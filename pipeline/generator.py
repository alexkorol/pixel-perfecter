"""Image generation orchestration for creating training pairs.

Takes pixel art sprites + unpixelization prompts and generates high-res
counterpart images via pluggable backends. The pixel art is the ground
truth target; the generated images are the training inputs.

Supported backends:
  - comfyui    — Local ComfyUI instance (best for volume)
  - a1111      — Local Automatic1111 / Forge WebUI
  - openai     — DALL-E 3 via OpenAI API
  - replicate  — Replicate API (Flux, SDXL, etc.)
  - manual     — Skip generation, expect pre-placed images

Set the backend via environment variables:
  GENERATOR_BACKEND=comfyui|a1111|openai|replicate|manual
  COMFYUI_HOST=http://localhost:8188
  A1111_HOST=http://localhost:7860
  OPENAI_API_KEY=...
  REPLICATE_API_TOKEN=...
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.config import InputStyle

logger = logging.getLogger(__name__)


@dataclass
class GeneratedImage:
    """A single generated high-res image for a training pair."""
    image: np.ndarray           # RGB numpy array
    style: InputStyle
    prompt: str
    sprite_path: str            # path to the source pixel art
    seed: int = 0
    model_name: str = ""
    generation_params: dict = None

    def __post_init__(self):
        if self.generation_params is None:
            self.generation_params = {}


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    width: int = 1024
    height: int = 1024
    num_images: int = 1         # images per prompt
    guidance_scale: float = 7.5
    num_steps: int = 30
    seed: Optional[int] = None  # None = random
    negative_prompt: str = (
        "pixel art, pixelated, mosaic, grid pattern, low resolution, "
        "blocky, jagged edges, retro game style, 8-bit, 16-bit, dithering"
    )
    model: str = ""             # backend-specific model name


# ---------------------------------------------------------------------------
# Backend: ComfyUI
# ---------------------------------------------------------------------------

def _generate_comfyui(
    prompt: str,
    config: GenerationConfig,
    sprite_img: Optional[np.ndarray] = None,
    host: Optional[str] = None,
) -> List[np.ndarray]:
    """Generate images using a local ComfyUI instance.

    Uses the /prompt API endpoint. Requires a workflow template
    that accepts text prompts. If sprite_img is provided, uploads
    it as a reference image for img2img workflows.
    """
    import urllib.request
    import urllib.error
    from io import BytesIO

    host = host or os.environ.get("COMFYUI_HOST", "http://localhost:8188")

    # Build a simple txt2img workflow
    # This is a minimal SDXL workflow — customize for your setup
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": config.seed or int(time.time()) % (2**32),
                "steps": config.num_steps,
                "cfg": config.guidance_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": config.model or "sd_xl_base_1.0.safetensors",
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": config.width,
                "height": config.height,
                "batch_size": config.num_images,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": config.negative_prompt,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "pipeline",
                "images": ["8", 0],
            },
        },
    }

    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prompt_id = data.get("prompt_id", "")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Could not connect to ComfyUI at {host}: {e}")

    # Poll for completion
    for _ in range(600):  # 10 min timeout
        time.sleep(1)
        try:
            status_req = urllib.request.Request(f"{host}/history/{prompt_id}")
            with urllib.request.urlopen(status_req, timeout=10) as resp:
                history = json.loads(resp.read().decode("utf-8"))
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                images_out = outputs.get("9", {}).get("images", [])
                if images_out:
                    break
        except Exception:
            continue
    else:
        raise TimeoutError("ComfyUI generation timed out")

    # Fetch generated images
    results = []
    for img_info in images_out:
        filename = img_info["filename"]
        subfolder = img_info.get("subfolder", "")
        img_url = f"{host}/view?filename={filename}&subfolder={subfolder}&type=output"
        with urllib.request.urlopen(img_url, timeout=30) as resp:
            img_data = resp.read()
        pil = Image.open(BytesIO(img_data)).convert("RGB")
        results.append(np.array(pil))

    return results


# ---------------------------------------------------------------------------
# Backend: Automatic1111 / Forge
# ---------------------------------------------------------------------------

def _generate_a1111(
    prompt: str,
    config: GenerationConfig,
    host: Optional[str] = None,
) -> List[np.ndarray]:
    """Generate images using the A1111/Forge txt2img API."""
    import urllib.request
    import base64
    from io import BytesIO

    host = host or os.environ.get("A1111_HOST", "http://localhost:7860")

    payload = json.dumps({
        "prompt": prompt,
        "negative_prompt": config.negative_prompt,
        "width": config.width,
        "height": config.height,
        "steps": config.num_steps,
        "cfg_scale": config.guidance_scale,
        "batch_size": config.num_images,
        "seed": config.seed or -1,
        "override_settings": {
            "sd_model_checkpoint": config.model,
        } if config.model else {},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/sdapi/v1/txt2img",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ConnectionError(f"A1111 API error: {e}")

    results = []
    for b64_img in data.get("images", []):
        img_bytes = base64.b64decode(b64_img)
        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        results.append(np.array(pil))

    return results


# ---------------------------------------------------------------------------
# Backend: OpenAI (DALL-E 3)
# ---------------------------------------------------------------------------

def _generate_openai(
    prompt: str,
    config: GenerationConfig,
) -> List[np.ndarray]:
    """Generate images using OpenAI's DALL-E 3 API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)

    # DALL-E 3 only supports 1 image at a time
    results = []
    for _ in range(config.num_images):
        response = client.images.generate(
            model=config.model or "dall-e-3",
            prompt=prompt,
            size=f"{config.width}x{config.height}",
            quality="hd",
            n=1,
        )
        image_url = response.data[0].url

        # Download the image
        import urllib.request
        from io import BytesIO
        with urllib.request.urlopen(image_url, timeout=60) as resp:
            img_data = resp.read()
        pil = Image.open(BytesIO(img_data)).convert("RGB")
        results.append(np.array(pil))

    return results


# ---------------------------------------------------------------------------
# Backend: Replicate
# ---------------------------------------------------------------------------

def _generate_replicate(
    prompt: str,
    config: GenerationConfig,
) -> List[np.ndarray]:
    """Generate images using the Replicate API."""
    try:
        import replicate
    except ImportError:
        raise ImportError("replicate package required. Install with: pip install replicate")

    model_name = config.model or "black-forest-labs/flux-1.1-pro"
    output = replicate.run(
        model_name,
        input={
            "prompt": prompt,
            "width": config.width,
            "height": config.height,
            "num_outputs": config.num_images,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_steps,
            "seed": config.seed,
        },
    )

    results = []
    import urllib.request
    from io import BytesIO
    for url in output:
        url_str = str(url)
        with urllib.request.urlopen(url_str, timeout=60) as resp:
            img_data = resp.read()
        pil = Image.open(BytesIO(img_data)).convert("RGB")
        results.append(np.array(pil))

    return results


# ---------------------------------------------------------------------------
# Unified generation function
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    config: Optional[GenerationConfig] = None,
    sprite_img: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> List[np.ndarray]:
    """Generate high-res images from a prompt using the configured backend.

    Args:
        prompt: The unpixelization prompt text.
        config: Generation configuration.
        sprite_img: Optional reference sprite for img2img.
        backend: Override backend selection.

    Returns:
        List of generated RGB numpy arrays.
    """
    if config is None:
        config = GenerationConfig()
    backend = backend or os.environ.get("GENERATOR_BACKEND", "comfyui")

    if backend == "comfyui":
        return _generate_comfyui(prompt, config, sprite_img)
    elif backend == "a1111":
        return _generate_a1111(prompt, config)
    elif backend == "openai":
        return _generate_openai(prompt, config)
    elif backend == "replicate":
        return _generate_replicate(prompt, config)
    elif backend == "manual":
        return []  # manual mode: images are pre-placed
    else:
        raise ValueError(f"Unknown generator backend: {backend}")


# ---------------------------------------------------------------------------
# Batch generation from manifest
# ---------------------------------------------------------------------------

def generate_pairs_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    config: Optional[GenerationConfig] = None,
    backend: Optional[str] = None,
    styles: Optional[List[InputStyle]] = None,
    rate_limit: float = 1.0,
    skip_existing: bool = True,
) -> int:
    """Generate training pair images for all sprites in a manifest.

    Reads the manifest written by the pipeline's prompt stage, generates
    high-res counterparts, and writes them alongside the prompts.

    Args:
        manifest_path: Path to the pipeline manifest.jsonl
        output_dir: Base output directory for generated images.
        config: Generation config.
        backend: Override backend.
        styles: Which styles to generate. None = all available.
        rate_limit: Seconds between API calls.
        skip_existing: Skip generation if output already exists.

    Returns:
        Number of images generated.
    """
    if config is None:
        config = GenerationConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    last_call = 0.0

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sprite_path = entry.get("sprite_path", "")
            stem = Path(sprite_path).stem

            # Find the prompt files for this sprite
            prompts_dir = manifest_path.parent / "prompts" / stem
            if not prompts_dir.exists():
                logger.warning("No prompts found for %s", stem)
                continue

            for prompt_file in sorted(prompts_dir.glob(f"{stem}_*.txt")):
                style_name = prompt_file.stem.replace(f"{stem}_", "")
                if style_name == "meta":
                    continue

                # Check style filter
                if styles:
                    try:
                        style = InputStyle(style_name)
                        if style not in styles:
                            continue
                    except ValueError:
                        continue

                # Check if output already exists
                gen_dir = output_dir / stem
                gen_dir.mkdir(parents=True, exist_ok=True)
                out_path = gen_dir / f"{stem}_{style_name}.png"
                if skip_existing and out_path.exists():
                    logger.debug("Skipping %s (exists)", out_path)
                    continue

                prompt_text = prompt_file.read_text(encoding="utf-8")

                # Rate limit
                elapsed = time.time() - last_call
                if elapsed < rate_limit:
                    time.sleep(rate_limit - elapsed)

                try:
                    # Load sprite for reference
                    sprite_pil = Image.open(sprite_path).convert("RGB")
                    sprite_arr = np.array(sprite_pil)

                    images = generate_image(
                        prompt_text,
                        config=config,
                        sprite_img=sprite_arr,
                        backend=backend,
                    )
                    last_call = time.time()

                    for i, img in enumerate(images):
                        if i == 0:
                            save_path = out_path
                        else:
                            save_path = gen_dir / f"{stem}_{style_name}_{i}.png"
                        Image.fromarray(img, "RGB").save(save_path)
                        count += 1
                        logger.info("Generated %s", save_path.name)

                except Exception as e:
                    logger.warning("Generation failed for %s/%s: %s",
                                 stem, style_name, e)

    logger.info("Generated %d images total", count)
    return count


# ---------------------------------------------------------------------------
# Manual pair collection helper
# ---------------------------------------------------------------------------

def collect_manual_pairs(
    manifest_path: Path,
    generated_dir: Path,
    output_dir: Path,
) -> int:
    """Collect manually-generated images into the pipeline structure.

    For workflows where you generate images outside the pipeline (e.g.
    using Midjourney, ChatGPT, or a local ComfyUI workflow), this function
    matches generated images to their source sprites and organizes them
    into the expected directory structure.

    Expected layout in generated_dir:
      <sprite_stem>_<style>.png
    or:
      <sprite_stem>/
        <style>.png

    Returns number of pairs collected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sprite_path = entry.get("sprite_path", "")
            stem = Path(sprite_path).stem

            # Look for matching generated images
            candidates = list(generated_dir.glob(f"{stem}_*.png"))
            candidates += list(generated_dir.glob(f"{stem}/*.png"))

            for img_path in candidates:
                pair_dir = output_dir / stem
                pair_dir.mkdir(parents=True, exist_ok=True)

                dest = pair_dir / img_path.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(img_path, dest)
                    count += 1
                    logger.info("Collected %s → %s", img_path.name, dest)

    logger.info("Collected %d manual pairs", count)
    return count
