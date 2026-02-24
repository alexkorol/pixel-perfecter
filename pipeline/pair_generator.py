"""Generate prompts and manage training pair creation.

For each cleaned pixel art asset, produces prompts to generate
"unpixelized" high-resolution versions in multiple styles.
These prompts are designed for tools like Nano Banana Pro, ChatGPT,
Gemini, etc.

The pixel art is ground truth (output). The generated high-res images
are training inputs. The model learns: detailed image + tags → pixel art.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import AssetTags, InputStyle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — one per input style
# ---------------------------------------------------------------------------

TEMPLATES: Dict[InputStyle, str] = {
    InputStyle.PHOTOREALISTIC: """\
You are looking at a small pixel art sprite from a 2D video game.
Your task is to imagine what real-world object, creature, or character
this sprite is meant to represent, then create a completely new, fully
photorealistic photograph-style image of that same subject.

SUBJECT: {description}

CRITICAL RULES:
- The output must be a COMPLETELY smooth, high-resolution image with NO
  pixel grid, NO checkerboard patterns, NO jagged edges, NO mosaic textures.
- Do NOT blend pixel art with realistic rendering. Do NOT preserve any
  pixel-level detail from the input. The output should look like a real
  photograph, not like an upscaled sprite.
- Match the subject matter exactly: same object/creature type, same pose,
  same colour scheme, same equipment/accessories, same facing direction.
- Use a plain solid neutral background.
- Show the full subject, nothing cropped.

Think of it this way: a pixel artist looked at a real photograph and
created this sprite from it. You are reconstructing that original photograph.

STYLE GUIDANCE:
- Studio lighting, crisp detail, neutral grey or black background.
- Material accuracy: metal reflects, cloth drapes, skin has texture.
- Keep detail level moderate — this should read clearly when downscaled.
""",

    InputStyle.DIGITAL_PAINTING: """\
You are looking at a small pixel art sprite from a 2D video game.
Your task is to identify what this sprite represents, then paint a
completely new, fully rendered digital painting of that exact same subject
as high-quality fantasy game concept art.

SUBJECT: {description}

CRITICAL RULES:
- The output must be a smooth, fully rendered digital painting. There must
  be ZERO pixel artifacts — no grid patterns, no checkerboard textures,
  no jagged staircase edges, no mosaic effects anywhere in the image.
- Do NOT upscale or "enhance" the pixel art. Do NOT blend pixel art
  aesthetics with painted aesthetics. Create a COMPLETELY NEW painting
  from scratch that depicts the same subject.
- Match precisely: same creature/object/character type, same pose and
  proportions, same facing direction, same colour palette, same equipment
  and accessories.
- Style: painterly fantasy concept art with visible brushwork, rich
  lighting, and material definition.
- Solid dark background, full subject visible.

Imagine a concept artist painted this subject first, and then a pixel
artist created the sprite from that painting. You are reconstructing
the original concept art.
""",

    InputStyle.CONCEPT_ART: """\
You are looking at a small pixel art sprite from a 2D video game.
Your task is to create a clean game concept art illustration of the
exact same subject, as if this concept art was the reference sheet
the pixel artist worked from.

SUBJECT: {description}

CRITICAL RULES:
- The output must be a perfectly smooth illustration. ZERO pixel
  artifacts — no grid, no checkerboard, no jagged edges, no dithering.
- Create a BRAND NEW illustration from scratch that depicts the same
  subject.
- Match precisely: same subject type, same shape and proportions,
  same angle/orientation, same colour palette.
- Style: clean digital illustration with crisp smooth outlines, flat
  colour fills with subtle gradients for dimension, minimal but
  effective shading.
- Centre the subject on a plain flat background.
- Do NOT add extra details, decorations, or context that aren't
  implied by the original sprite.

Think of it this way: an illustrator drew this as a concept art
reference sheet, and a pixel artist referenced that drawing to make
the sprite. You are recreating the illustrator's original.
""",

    InputStyle.ANIME: """\
You are looking at a small pixel art sprite from a 2D video game.
Your task is to create an anime/manga-style illustration of the exact
same subject, as if this anime art was the character design reference
the pixel artist worked from.

SUBJECT: {description}

CRITICAL RULES:
- The output must be a clean anime-style illustration with NO pixel
  artifacts — no grid, no checkerboard, no jagged edges.
- Create a BRAND NEW illustration. Do NOT upscale the pixel art.
- Match: same subject, same pose, same colours, same facing direction,
  same equipment.
- Anime style: clean lines, cel-shaded, expressive but accurate to the
  source sprite's design.
- Flat solid background, full subject visible.
""",

    InputStyle.VECTOR: """\
You are looking at a pixel art game icon or item sprite.
Create a completely new, clean, high-resolution vector-style digital
illustration of the exact same object.

SUBJECT: {description}

CRITICAL RULES:
- The output must be a perfectly smooth vector-style illustration. ZERO
  pixel artifacts — no grid, no checkerboard, no jagged edges.
- Create a BRAND NEW illustration from scratch.
- Match: same object type, same shape and proportions, same
  angle/orientation, same colour palette.
- Style: clean vector art with crisp outlines, flat colour fills, subtle
  gradients for dimension. Minimal shading.
- Centre the object on a plain flat background.
- Do NOT add extra elements not implied by the source sprite.
""",
}


# ---------------------------------------------------------------------------
# Prompt modifiers — appended based on tags
# ---------------------------------------------------------------------------

def _build_modifiers(tags: AssetTags) -> str:
    """Build prompt modifier string from asset tags."""
    mods = []

    # Proportions guidance
    mods.append(
        "PROPORTIONS: Maintain the exact silhouette and proportions from "
        "the pixel art — do not 'correct' anatomy to realistic standards. "
        "If the sprite has chibi/exaggerated proportions, preserve them."
    )

    # Detail level based on grid size
    if tags.grid_size <= 16:
        mods.append(
            "DETAIL LEVEL: Keep detail moderate-to-low. This should read "
            f"clearly when downscaled to {tags.grid_size}x{tags.grid_size} pixels."
        )
    elif tags.grid_size <= 32:
        mods.append(
            "DETAIL LEVEL: Moderate detail. Should remain recognisable when "
            f"downscaled to roughly {tags.grid_size}x{tags.grid_size}."
        )
    else:
        mods.append(
            "DETAIL LEVEL: Can include finer detail — the target sprite "
            f"resolution is {tags.grid_size}x{tags.grid_size} pixels."
        )

    # Palette
    if tags.palette_count <= 16:
        hex_str = ", ".join(f"#{c}" for c in tags.palette_hex[:16])
        mods.append(
            f"COLOUR PALETTE: Use ONLY these colours (or very close "
            f"analogues): {hex_str}. Do not introduce new hues."
        )
    elif tags.palette_name:
        mods.append(
            f"COLOUR PALETTE: Colour scheme should align with the "
            f"'{tags.palette_name}' palette. Stay within its hue range."
        )

    # Background
    mods.append(
        "BACKGROUND: Pure flat single-colour background. No gradients, "
        "no ground plane, no shadows cast on the background."
    )

    return "\n\n".join(mods)


# ---------------------------------------------------------------------------
# VLM captioning interface (pluggable)
# ---------------------------------------------------------------------------

@dataclass
class SpriteCaption:
    """Description of a pixel art sprite, either from metadata or VLM."""
    short_description: str       # one-line (e.g. "a knight in blue armour")
    detailed_description: str    # multi-sentence with all visible details
    source: str = "manual"       # "manual", "vlm", "game_data"


def caption_from_game_data(sprite_name: str, game_text: str) -> SpriteCaption:
    """Create a caption from existing game metadata (DCSS, etc.)."""
    return SpriteCaption(
        short_description=sprite_name,
        detailed_description=game_text,
        source="game_data",
    )


def caption_placeholder(sprite_name: str) -> SpriteCaption:
    """Placeholder caption when no VLM or game data is available.

    In production, this would call a vision model to describe the sprite.
    """
    return SpriteCaption(
        short_description=sprite_name,
        detailed_description=(
            f"A pixel art game sprite: {sprite_name}. "
            "Describe all visible details including pose, equipment, "
            "colours, and any distinctive features."
        ),
        source="placeholder",
    )


# VLM integration stub — implement with your preferred VLM
def caption_with_vlm(image_path: str, model: str = "default") -> SpriteCaption:
    """Generate a caption using a vision-language model.

    This is a stub. Implement with your VLM of choice:
    - Local: LLaVA, CogVLM, InternVL
    - API: GPT-4V, Gemini Pro Vision, Claude

    The VLM should be prompted:
      "Describe this pixel art sprite in detail. Include: what it depicts,
       the pose/orientation, all equipment/accessories, colour palette,
       and any distinctive features. Be precise — this description will
       be used to recreate the subject as a high-resolution illustration."
    """
    raise NotImplementedError(
        "VLM captioning not yet configured. Set up a vision model or "
        "use caption_from_game_data() / manual captions."
    )


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationPrompt:
    """A ready-to-use prompt for generating an unpixelized training input."""
    style: InputStyle
    prompt_text: str
    sprite_path: str
    tags: AssetTags
    caption: SpriteCaption


def generate_prompts(
    sprite_path: str,
    tags: AssetTags,
    caption: SpriteCaption,
    styles: Optional[List[InputStyle]] = None,
) -> List[GenerationPrompt]:
    """Generate unpixelization prompts for a single sprite.

    Args:
        sprite_path: Path to the pixel art sprite (for reference).
        tags: Auto-detected tags for this sprite.
        caption: Description of what the sprite depicts.
        styles: Which input styles to generate. Defaults to all.

    Returns:
        List of GenerationPrompt, one per style.
    """
    if styles is None:
        styles = list(InputStyle)

    modifiers = _build_modifiers(tags)
    prompts = []

    for style in styles:
        template = TEMPLATES.get(style)
        if template is None:
            logger.warning("No template for style %s", style)
            continue

        text = template.format(description=caption.detailed_description)
        text = text.strip() + "\n\n" + modifiers

        prompts.append(GenerationPrompt(
            style=style,
            prompt_text=text,
            sprite_path=sprite_path,
            tags=tags,
            caption=caption,
        ))

    return prompts


def save_prompts(
    prompts: List[GenerationPrompt],
    output_dir: Path,
    sprite_stem: str,
) -> List[Path]:
    """Save generated prompts as text files + a metadata JSON.

    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for prompt in prompts:
        style_name = prompt.style.value
        txt_path = output_dir / f"{sprite_stem}_{style_name}.txt"
        txt_path.write_text(prompt.prompt_text, encoding="utf-8")
        saved.append(txt_path)

    # save metadata
    meta_path = output_dir / f"{sprite_stem}_meta.json"
    meta = {
        "sprite_path": prompts[0].sprite_path if prompts else "",
        "caption": {
            "short": prompts[0].caption.short_description if prompts else "",
            "detailed": prompts[0].caption.detailed_description if prompts else "",
            "source": prompts[0].caption.source if prompts else "",
        },
        "tags": prompts[0].tags.to_dict() if prompts else {},
        "styles": [p.style.value for p in prompts],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    saved.append(meta_path)

    return saved


# ---------------------------------------------------------------------------
# Batch prompt generation
# ---------------------------------------------------------------------------

def batch_generate_prompts(
    manifest_path: Path,
    output_dir: Path,
    styles: Optional[List[InputStyle]] = None,
) -> int:
    """Generate prompts for all sprites in a manifest.

    Manifest format: JSONL with fields:
        sprite_path, tags (dict), caption_short, caption_detailed, caption_source

    Returns number of sprites processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            sprite_path = entry["sprite_path"]
            tags = AssetTags.from_dict(entry["tags"])
            caption = SpriteCaption(
                short_description=entry.get("caption_short", ""),
                detailed_description=entry.get("caption_detailed", ""),
                source=entry.get("caption_source", "manual"),
            )

            stem = Path(sprite_path).stem
            prompts = generate_prompts(sprite_path, tags, caption, styles)
            save_prompts(prompts, output_dir / stem, stem)
            count += 1

    logger.info("Generated prompts for %d sprites", count)
    return count
