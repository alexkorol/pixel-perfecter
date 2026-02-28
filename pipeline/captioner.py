"""VLM-based sprite captioning for the pixel art pipeline.

Describes pixel art sprites using vision-language models so that
unpixelization prompts contain accurate subject descriptions rather
than placeholder text.

Supported backends:
  - Claude (Anthropic API) — default, best quality
  - OpenAI (GPT-4o / GPT-4 Vision)
  - Local  (ollama with llava / bakllava)
  - Manual (reads from a sidecar JSON)

Set the backend via environment variables:
  CAPTIONER_BACKEND=claude|openai|local|manual
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...
  OLLAMA_HOST=http://localhost:11434  (for local)
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Re-use the SpriteCaption dataclass from pair_generator
from pipeline.pair_generator import SpriteCaption


# ---------------------------------------------------------------------------
# System prompt for all VLM backends
# ---------------------------------------------------------------------------

CAPTION_SYSTEM_PROMPT = """\
You are an expert at analyzing pixel art sprites. You will be shown a small \
pixel art image (possibly at 1x native resolution, possibly upscaled). \
Your job is to describe it precisely so that an image generation model can \
recreate the same subject as a high-resolution illustration.

Respond with ONLY a JSON object (no markdown fences):
{
  "short": "<one-line description, e.g. 'a knight in blue armour holding a sword'>",
  "detailed": "<3-5 sentences covering: subject identity, pose/stance, facing direction, all equipment/accessories, dominant colours, any distinctive features, overall silhouette shape>"
}

Be PRECISE about what you see. Do not hallucinate details that are not \
visible. If the sprite is too small or abstract to identify, say so. \
Focus on visual details that would help reproduce the subject faithfully."""


def _image_to_base64(img: np.ndarray) -> str:
    """Encode a numpy image as a base64 PNG string."""
    from io import BytesIO
    if img.shape[2] == 4:
        pil = Image.fromarray(img, "RGBA")
    else:
        pil = Image.fromarray(img, "RGB")
    # upscale tiny sprites so the VLM can see them
    w, h = pil.size
    if max(w, h) < 64:
        scale = max(1, 64 // max(w, h))
        pil = pil.resize((w * scale, h * scale), Image.NEAREST)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_caption_json(text: str) -> Tuple[str, str]:
    """Extract short and detailed descriptions from VLM JSON response."""
    # strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        d = json.loads(text)
        return d.get("short", ""), d.get("detailed", "")
    except json.JSONDecodeError:
        # fallback: use the whole response as the detailed description
        logger.warning("Failed to parse VLM response as JSON, using raw text")
        # try to extract short from first sentence
        sentences = text.split(".")
        short = sentences[0].strip() if sentences else text[:80]
        return short, text


# ---------------------------------------------------------------------------
# Backend: Claude (Anthropic)
# ---------------------------------------------------------------------------

def _caption_claude(
    img: np.ndarray,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
) -> SpriteCaption:
    """Caption a sprite using the Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for Claude captioning. "
            "Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    b64 = _image_to_base64(img)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": CAPTION_SYSTEM_PROMPT,
                        },
                    ],
                }],
            )
            text = response.content[0].text
            short, detailed = _parse_caption_json(text)
            return SpriteCaption(
                short_description=short,
                detailed_description=detailed,
                source="claude",
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Claude API error (attempt %d): %s, retrying in %ds",
                             attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------

def _caption_openai(
    img: np.ndarray,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> SpriteCaption:
    """Caption a sprite using the OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package required for OpenAI captioning. "
            "Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)
    b64 = _image_to_base64(img)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": CAPTION_SYSTEM_PROMPT,
                        },
                    ],
                }],
            )
            text = response.choices[0].message.content
            short, detailed = _parse_caption_json(text)
            return SpriteCaption(
                short_description=short,
                detailed_description=detailed,
                source="openai",
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("OpenAI API error (attempt %d): %s, retrying in %ds",
                             attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Backend: Local (Ollama)
# ---------------------------------------------------------------------------

def _caption_local(
    img: np.ndarray,
    model: str = "llava",
    host: Optional[str] = None,
) -> SpriteCaption:
    """Caption a sprite using a local Ollama model."""
    import urllib.request
    import urllib.error

    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    b64 = _image_to_base64(img)

    payload = json.dumps({
        "model": model,
        "prompt": CAPTION_SYSTEM_PROMPT,
        "images": [b64],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        short, detailed = _parse_caption_json(text)
        return SpriteCaption(
            short_description=short,
            detailed_description=detailed,
            source=f"local:{model}",
        )
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not connect to Ollama at {host}: {e}"
        )


# ---------------------------------------------------------------------------
# Backend: Manual (sidecar JSON)
# ---------------------------------------------------------------------------

def _caption_manual(source_path: str) -> SpriteCaption:
    """Read caption from a sidecar JSON file next to the image.

    Looks for <image_stem>_caption.json or <image_stem>.json with:
    {"short": "...", "detailed": "..."}
    """
    p = Path(source_path)
    candidates = [
        p.with_name(f"{p.stem}_caption.json"),
        p.with_suffix(".json"),
    ]
    for cand in candidates:
        if cand.exists():
            data = json.loads(cand.read_text(encoding="utf-8"))
            return SpriteCaption(
                short_description=data.get("short", p.stem),
                detailed_description=data.get("detailed", ""),
                source="manual",
            )
    raise FileNotFoundError(
        f"No caption sidecar found for {source_path}. "
        f"Expected one of: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Unified captioning function
# ---------------------------------------------------------------------------

def caption_sprite(
    img: np.ndarray,
    source_path: str = "",
    backend: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> SpriteCaption:
    """Caption a pixel art sprite using the configured VLM backend.

    Args:
        img: RGB or RGBA numpy array (1x or upscaled).
        source_path: Path to the source image file (for manual/caching).
        backend: Override backend (claude|openai|local|manual).
            Defaults to CAPTIONER_BACKEND env var, then "claude".
        model: Override model name for the backend.
        cache_dir: Directory to cache captions. If set, captions are
            written as JSON sidecars and reused on subsequent runs.

    Returns:
        SpriteCaption with short and detailed descriptions.
    """
    backend = backend or os.environ.get("CAPTIONER_BACKEND", "claude")

    # Check cache first
    if cache_dir and source_path:
        cache_path = Path(cache_dir) / f"{Path(source_path).stem}_caption.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.debug("Using cached caption for %s", source_path)
            return SpriteCaption(
                short_description=data["short"],
                detailed_description=data["detailed"],
                source=data.get("source", "cached"),
            )

    if backend == "manual":
        caption = _caption_manual(source_path)
    elif backend == "claude":
        caption = _caption_claude(img, model=model or "claude-sonnet-4-20250514")
    elif backend == "openai":
        caption = _caption_openai(img, model=model or "gpt-4o")
    elif backend == "local":
        caption = _caption_local(img, model=model or "llava")
    else:
        raise ValueError(f"Unknown captioning backend: {backend}")

    # Cache the result
    if cache_dir and source_path:
        cache_path = Path(cache_dir) / f"{Path(source_path).stem}_caption.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "short": caption.short_description,
            "detailed": caption.detailed_description,
            "source": caption.source,
        }, indent=2), encoding="utf-8")
        logger.debug("Cached caption for %s", source_path)

    return caption


# ---------------------------------------------------------------------------
# Batch captioning
# ---------------------------------------------------------------------------

def batch_caption(
    sprite_dir: Path,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    rate_limit: float = 0.5,
) -> List[Tuple[str, SpriteCaption]]:
    """Caption all sprites in a directory.

    Args:
        sprite_dir: Directory containing sprite images.
        backend: VLM backend to use.
        model: Model name override.
        cache_dir: Cache directory for captions.
        rate_limit: Minimum seconds between API calls.

    Returns:
        List of (filename, SpriteCaption).
    """
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    results = []
    last_call = 0.0

    for fpath in sorted(Path(sprite_dir).iterdir()):
        if fpath.suffix.lower() not in extensions:
            continue

        try:
            pil = Image.open(fpath)
            if pil.mode == "RGBA":
                img = np.array(pil)
            elif pil.mode == "P":
                img = np.array(pil.convert("RGBA"))
            else:
                img = np.array(pil.convert("RGB"))
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            continue

        # rate limiting for API backends
        if backend not in ("manual", None) or (backend is None and
                os.environ.get("CAPTIONER_BACKEND", "claude") != "manual"):
            elapsed = time.time() - last_call
            if elapsed < rate_limit:
                time.sleep(rate_limit - elapsed)

        try:
            caption = caption_sprite(
                img,
                source_path=str(fpath),
                backend=backend,
                model=model,
                cache_dir=cache_dir,
            )
            results.append((fpath.name, caption))
            last_call = time.time()
            logger.info("Captioned %s: %s", fpath.name, caption.short_description)
        except Exception as e:
            logger.warning("Failed to caption %s: %s", fpath.name, e)

    return results
