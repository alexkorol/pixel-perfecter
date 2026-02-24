"""Ingest DCSS tiles into the pipeline.

DCSS tiles are already clean pixel art at native 1x resolution (32x32
or 32x48), so they skip the sheet-splitting and cleaning stages. We go
straight to tagging + prompt generation.

The file paths and directory structure provide rich metadata:
  mon/demons/shadow_demon.png  → subject:monster, subcategory:demons
  item/weapon/long_sword.png   → subject:item, subcategory:weapon
  player/base/human_m.png      → subject:character, subcategory:base
  dngn/floor/grey_dirt0.png    → subject:tile, subcategory:floor

Usage:
    python -m pipeline.ingest_dcss <tiles-dir> -o <output-dir>
    python -m pipeline.ingest_dcss data/dcss-tiles/releases/Nov-2015 -o data/dcss
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pipeline.config import (
    AssetTags,
    FacingDirection,
    InputStyle,
    OutlineType,
    PipelineConfig,
    ShadingStyle,
    SubjectType,
)
from pipeline.tagger import auto_tag
from pipeline.pair_generator import (
    SpriteCaption,
    generate_prompts,
    save_prompts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCSS path → metadata mapping
# ---------------------------------------------------------------------------

# Top-level category → SubjectType
CATEGORY_MAP: Dict[str, SubjectType] = {
    "mon": SubjectType.MONSTER,
    "player": SubjectType.CHARACTER,
    "item": SubjectType.ITEM,
    "dngn": SubjectType.TILE,
    "effect": SubjectType.EFFECT,
    "gui": SubjectType.UI,
    "misc": SubjectType.ICON,
}

# Subcategories that hint at facing direction
FACING_HINTS: Dict[str, FacingDirection] = {
    "north": FacingDirection.BACK,
    "south": FacingDirection.FRONT,
    "east": FacingDirection.RIGHT,
    "west": FacingDirection.LEFT,
}


def _filename_to_description(filepath: str) -> str:
    """Convert a DCSS tile filename/path into a natural language description.

    Examples:
        mon/demons/shadow_demon.png → "a shadow demon (demons category)"
        item/weapon/long_sword.png → "a long sword (weapon)"
        player/base/human_m.png → "a male human character (base sprite)"
        dngn/wall/brick_dark3.png → "dark brick wall tile (variant 3)"
    """
    parts = Path(filepath).parts
    stem = Path(filepath).stem

    # strip trailing numbers (variants like brick_dark3 → brick_dark)
    variant = ""
    match = re.match(r"^(.+?)(\d+)$", stem)
    if match:
        stem_clean = match.group(1).rstrip("_")
        variant = f" (variant {match.group(2)})"
    else:
        stem_clean = stem

    # convert underscores to spaces
    name = stem_clean.replace("_", " ")

    # determine category context
    category = parts[0] if len(parts) > 1 else ""
    subcategory = parts[1] if len(parts) > 2 else ""

    # build description based on category
    if category == "mon":
        article = "an" if name[0].lower() in "aeiou" else "a"
        desc = f"{article} {name}"
        if subcategory:
            desc += f", a creature from the {subcategory.replace('_', ' ')} family"
        desc += ". This is a monster sprite from a fantasy roguelike dungeon crawler game."

    elif category == "player":
        # detect gender hints
        if name.endswith(" m"):
            name = name[:-2]
            gender = "male"
        elif name.endswith(" f"):
            name = name[:-2]
            gender = "female"
        else:
            gender = ""

        if subcategory == "base":
            desc = f"a {gender + ' ' if gender else ''}{name} character, full body sprite"
        elif subcategory in ("cloak", "helm", "body", "boots", "hand1", "hand2",
                              "leg", "arm", "hair", "beard", "wing", "drcwing",
                              "drchead", "felid", "octopode"):
            desc = f"{name} equipment overlay ({subcategory}) for a player character"
        else:
            desc = f"a {gender + ' ' if gender else ''}{name} ({subcategory}) player sprite"
        desc += ". From a fantasy roguelike RPG."

    elif category == "item":
        article = "an" if name[0].lower() in "aeiou" else "a"
        if subcategory:
            desc = f"{article} {name}, a {subcategory.replace('_', ' ')} item"
        else:
            desc = f"{article} {name} item"
        desc += ". A game item icon from a fantasy roguelike."

    elif category == "dngn":
        if subcategory == "floor":
            desc = f"{name} floor tile for a dungeon environment"
        elif subcategory == "wall":
            desc = f"{name} wall tile for a dungeon environment"
        elif subcategory == "doors":
            desc = f"a {name} door tile"
        elif subcategory == "altars":
            desc = f"an altar to {name}"
        elif subcategory in ("gateways", "traps", "shops"):
            desc = f"a {name} {subcategory.rstrip('s')}"
        else:
            desc = f"{name} dungeon feature ({subcategory})" if subcategory else f"{name} dungeon tile"
        desc += ". A tileable game environment sprite."

    elif category == "effect":
        desc = f"{name} visual effect sprite"
        desc += ". A game effect animation frame."

    elif category == "gui":
        desc = f"{name} user interface element"

    elif category == "misc":
        desc = f"{name} miscellaneous game sprite"

    else:
        desc = f"{name} game sprite"

    if variant:
        desc += variant

    return desc


def _detect_facing_from_path(filepath: str) -> FacingDirection:
    """Attempt to detect facing direction from filename hints."""
    stem = Path(filepath).stem.lower()
    for hint, direction in FACING_HINTS.items():
        if hint in stem:
            return direction
    # DCSS sprites are mostly front-facing
    parts = Path(filepath).parts
    category = parts[0] if len(parts) > 1 else ""
    if category in ("mon", "player"):
        return FacingDirection.FRONT
    if category == "dngn":
        return FacingDirection.TOP_DOWN
    if category == "item":
        return FacingDirection.FRONT
    return FacingDirection.NA


def _detect_animation_frame(filepath: str) -> str:
    """Detect if this is an animation frame from filename."""
    stem = Path(filepath).stem
    # patterns like: walk_1, idle_0, attack_2
    match = re.search(r"(walk|idle|attack|cast|hit|die|run|fly)[-_]?(\d+)", stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return ""


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

# Skip tiles that are too small, non-standard, or UI elements
SKIP_CATEGORIES = {"UNUSED"}
MIN_TILE_SIZE = 16  # skip anything smaller than 16x16


def should_skip(filepath: str, img_size: Tuple[int, int]) -> bool:
    """Check if a tile should be skipped."""
    parts = Path(filepath).parts
    if parts[0] in SKIP_CATEGORIES:
        return True

    w, h = img_size
    if w < MIN_TILE_SIZE or h < MIN_TILE_SIZE:
        return True

    # skip very large images (likely composite sheets, not individual tiles)
    if w > 128 or h > 128:
        return True

    return False


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_dcss(
    tiles_dir: Path,
    output_dir: Path,
    styles: Optional[List[InputStyle]] = None,
    generate_prompt_files: bool = True,
    categories: Optional[List[str]] = None,
) -> int:
    """Ingest DCSS tiles into the pipeline.

    Args:
        tiles_dir: Path to DCSS tiles release directory
            (e.g., data/dcss-tiles/releases/Nov-2015)
        output_dir: Pipeline output directory.
        styles: Input styles for prompt generation. Defaults to photo + painting + concept.
        generate_prompt_files: Write prompt .txt files.
        categories: Only process these categories (e.g. ["mon", "item"]).
            Defaults to all.

    Returns:
        Number of tiles processed.
    """
    if styles is None:
        styles = [InputStyle.PHOTOREALISTIC, InputStyle.DIGITAL_PAINTING,
                  InputStyle.CONCEPT_ART]

    cleaned_dir = output_dir / "cleaned"
    prompts_dir = output_dir / "prompts"
    manifest_path = output_dir / "manifest.jsonl"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    skipped = 0
    processed = 0

    # walk the tiles directory
    all_pngs = sorted(tiles_dir.rglob("*.png"))
    logger.info("Found %d PNG files in %s", len(all_pngs), tiles_dir)

    for png_path in all_pngs:
        rel_path = png_path.relative_to(tiles_dir)
        parts = rel_path.parts

        # filter by category
        if categories and parts[0] not in categories:
            continue

        # load image
        try:
            pil = Image.open(png_path).convert("RGBA")
        except Exception as e:
            logger.warning("Failed to load %s: %s", rel_path, e)
            continue

        if should_skip(str(rel_path), pil.size):
            skipped += 1
            continue

        img = np.array(pil)
        w, h = pil.size

        # DCSS tiles are already 1x — no cleaning needed
        # save a copy to cleaned dir (preserving directory structure)
        out_subdir = cleaned_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_1x = out_subdir / f"{rel_path.stem}_1x.png"
        pil.save(out_1x)

        # create NN-upscaled version for training
        scale = max(1, 256 // max(w, h))
        img_up = cv2.resize(img, (w * scale, h * scale),
                            interpolation=cv2.INTER_NEAREST)
        out_up = out_subdir / f"{rel_path.stem}_up.png"
        Image.fromarray(img_up, "RGBA").save(out_up)

        # auto-tag
        tags = auto_tag(img, grid_size=32, source_set="dcss")

        # enrich with path-derived metadata
        if parts[0] in CATEGORY_MAP:
            tags.subject = CATEGORY_MAP[parts[0]]
        tags.facing = _detect_facing_from_path(str(rel_path))
        tags.animation_frame = _detect_animation_frame(str(rel_path))

        # generate description from filename
        description = _filename_to_description(str(rel_path))
        caption = SpriteCaption(
            short_description=rel_path.stem.replace("_", " "),
            detailed_description=description,
            source="dcss_path",
        )

        # generate prompts
        if generate_prompt_files:
            prompts = generate_prompts(str(out_1x), tags, caption, styles)
            prompt_subdir = prompts_dir / rel_path.parent / rel_path.stem
            save_prompts(prompts, prompt_subdir, rel_path.stem)

        # manifest entry
        manifest_entries.append({
            "sprite_path": str(out_1x),
            "sprite_upscaled_path": str(out_up),
            "source_image": str(png_path),
            "dcss_path": str(rel_path),
            "tags": tags.to_dict(),
            "caption_short": caption.short_description,
            "caption_detailed": caption.detailed_description,
            "caption_source": caption.source,
            "cell_size": 32,
            "metrics": {"method": "dcss_native", "percent_diff_core": 0.0},
        })

        processed += 1
        if processed % 200 == 0:
            logger.info("  Processed %d tiles...", processed)

    # write manifest
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info("DCSS ingestion complete:")
    logger.info("  Processed: %d tiles", processed)
    logger.info("  Skipped:   %d tiles", skipped)
    logger.info("  Manifest:  %s", manifest_path)

    # summary stats
    from collections import Counter
    subjects = Counter(e["tags"]["subject"] for e in manifest_entries)
    outlines = Counter(e["tags"]["outline"] for e in manifest_entries)
    logger.info("  Subjects: %s", dict(subjects))
    logger.info("  Outlines: %s", dict(outlines))

    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest DCSS tiles into the pixelization pipeline.",
    )
    parser.add_argument("tiles_dir",
                        help="Path to DCSS tiles release (e.g. data/dcss-tiles/releases/Nov-2015)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only process these categories (e.g. mon item player)")
    parser.add_argument("--styles", default=None,
                        help="Comma-separated generation styles")
    parser.add_argument("--no-prompts", action="store_true",
                        help="Skip prompt file generation")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    styles = None
    if args.styles:
        styles = [InputStyle(s.strip()) for s in args.styles.split(",")]

    count = ingest_dcss(
        tiles_dir=Path(args.tiles_dir),
        output_dir=Path(args.output),
        styles=styles,
        generate_prompt_files=not args.no_prompts,
        categories=args.categories,
    )

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
