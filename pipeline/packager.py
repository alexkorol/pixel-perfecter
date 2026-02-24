"""Package cleaned sprites + generated pairs into a training-ready dataset.

Final dataset format:
  data/pairs/
    manifest.jsonl          # master manifest
    train/
      0001/
        input.png           # high-res generated image (1024x1024)
        target.png          # pixel art at training resolution (NN upscaled)
        target_1x.png       # pixel art at native 1x resolution
        meta.json           # tags, caption, metrics
      0002/
        ...
    val/
    test/
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pipeline.config import AssetTags

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """A single training pair: high-res input → pixel art target."""
    pair_id: str
    input_path: str             # path to high-res generated image
    target_1x_path: str         # path to pixel art at 1x resolution
    target_upscaled_path: str   # path to NN-upscaled pixel art
    tags: AssetTags
    caption_short: str = ""
    caption_detailed: str = ""
    input_style: str = ""       # "photorealistic", "digital_painting", etc.
    quality_score: float = 0.0  # 0-1, from QC pass
    split: str = "train"        # "train", "val", "test"


@dataclass
class DatasetStats:
    """Summary statistics for a packaged dataset."""
    total_pairs: int = 0
    train_pairs: int = 0
    val_pairs: int = 0
    test_pairs: int = 0
    grid_size_distribution: Dict[int, int] = field(default_factory=dict)
    style_distribution: Dict[str, int] = field(default_factory=dict)
    outline_distribution: Dict[str, int] = field(default_factory=dict)
    palette_size_distribution: Dict[str, int] = field(default_factory=dict)
    mean_quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_pairs": self.total_pairs,
            "train_pairs": self.train_pairs,
            "val_pairs": self.val_pairs,
            "test_pairs": self.test_pairs,
            "grid_size_distribution": self.grid_size_distribution,
            "style_distribution": self.style_distribution,
            "outline_distribution": self.outline_distribution,
            "palette_size_distribution": self.palette_size_distribution,
            "mean_quality_score": self.mean_quality_score,
        }


def assign_splits(
    pairs: List[TrainingPair],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> List[TrainingPair]:
    """Assign train/val/test splits to pairs.

    Groups by sprite (same target) to prevent data leakage — all style
    variants of the same sprite go to the same split.
    """
    rng = random.Random(seed)

    # group by target sprite
    sprite_groups: Dict[str, List[int]] = {}
    for i, pair in enumerate(pairs):
        key = pair.target_1x_path
        if key not in sprite_groups:
            sprite_groups[key] = []
        sprite_groups[key].append(i)

    sprite_keys = list(sprite_groups.keys())
    rng.shuffle(sprite_keys)

    n_sprites = len(sprite_keys)
    n_val = max(1, int(n_sprites * val_ratio))
    n_test = max(1, int(n_sprites * test_ratio))

    val_sprites = set(sprite_keys[:n_val])
    test_sprites = set(sprite_keys[n_val:n_val + n_test])

    for pair in pairs:
        key = pair.target_1x_path
        if key in val_sprites:
            pair.split = "val"
        elif key in test_sprites:
            pair.split = "test"
        else:
            pair.split = "train"

    return pairs


def validate_pair(
    input_path: str,
    target_path: str,
    min_input_size: int = 128,
    min_target_size: int = 4,
) -> Tuple[bool, str]:
    """Validate a single training pair.

    Returns (is_valid, reason).
    """
    # check files exist
    if not Path(input_path).exists():
        return False, f"Input file missing: {input_path}"
    if not Path(target_path).exists():
        return False, f"Target file missing: {target_path}"

    try:
        inp = Image.open(input_path)
        tgt = Image.open(target_path)
    except Exception as e:
        return False, f"Failed to load images: {e}"

    # check dimensions
    if min(inp.size) < min_input_size:
        return False, f"Input too small: {inp.size}"
    if min(tgt.size) < min_target_size:
        return False, f"Target too small: {tgt.size}"

    # check that input is actually high-res relative to target
    if inp.size[0] < tgt.size[0] * 2:
        return False, (
            f"Input ({inp.size}) not sufficiently larger than "
            f"target ({tgt.size})"
        )

    return True, "ok"


def compute_dataset_stats(pairs: List[TrainingPair]) -> DatasetStats:
    """Compute summary statistics for the dataset."""
    stats = DatasetStats()
    stats.total_pairs = len(pairs)

    quality_scores = []

    for pair in pairs:
        if pair.split == "train":
            stats.train_pairs += 1
        elif pair.split == "val":
            stats.val_pairs += 1
        else:
            stats.test_pairs += 1

        gs = pair.tags.grid_size
        stats.grid_size_distribution[gs] = stats.grid_size_distribution.get(gs, 0) + 1

        style = pair.input_style
        stats.style_distribution[style] = stats.style_distribution.get(style, 0) + 1

        outline = pair.tags.outline.value if hasattr(pair.tags.outline, 'value') else str(pair.tags.outline)
        stats.outline_distribution[outline] = stats.outline_distribution.get(outline, 0) + 1

        # bucket palette sizes
        pc = pair.tags.palette_count
        if pc <= 4:
            bucket = "1-4"
        elif pc <= 8:
            bucket = "5-8"
        elif pc <= 16:
            bucket = "9-16"
        elif pc <= 32:
            bucket = "17-32"
        elif pc <= 64:
            bucket = "33-64"
        else:
            bucket = "65+"
        stats.palette_size_distribution[bucket] = (
            stats.palette_size_distribution.get(bucket, 0) + 1
        )

        quality_scores.append(pair.quality_score)

    if quality_scores:
        stats.mean_quality_score = sum(quality_scores) / len(quality_scores)

    return stats


def write_manifest(pairs: List[TrainingPair], manifest_path: Path):
    """Write the master JSONL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for pair in pairs:
            entry = {
                "pair_id": pair.pair_id,
                "input_path": pair.input_path,
                "target_1x_path": pair.target_1x_path,
                "target_upscaled_path": pair.target_upscaled_path,
                "tags": pair.tags.to_dict(),
                "tag_string": pair.tags.to_tag_string(),
                "caption_short": pair.caption_short,
                "caption_detailed": pair.caption_detailed,
                "input_style": pair.input_style,
                "quality_score": pair.quality_score,
                "split": pair.split,
            }
            f.write(json.dumps(entry) + "\n")

    logger.info("Wrote manifest with %d pairs to %s", len(pairs), manifest_path)


def package_dataset(
    pairs: List[TrainingPair],
    output_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetStats:
    """Package pairs into the final dataset directory structure.

    Creates the directory layout, copies/symlinks files, writes manifest
    and stats.

    Returns DatasetStats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # assign splits
    pairs = assign_splits(pairs, val_ratio, test_ratio, seed)

    # write manifest
    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(pairs, manifest_path)

    # write stats
    stats = compute_dataset_stats(pairs)
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))

    # organize into split directories
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(exist_ok=True)

    for pair in pairs:
        pair_dir = output_dir / pair.split / pair.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        # write pair metadata
        meta = {
            "pair_id": pair.pair_id,
            "tags": pair.tags.to_dict(),
            "tag_string": pair.tags.to_tag_string(),
            "caption_short": pair.caption_short,
            "caption_detailed": pair.caption_detailed,
            "input_style": pair.input_style,
            "quality_score": pair.quality_score,
        }
        (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "Packaged dataset: %d train / %d val / %d test",
        stats.train_pairs, stats.val_pairs, stats.test_pairs,
    )

    return stats


def load_manifest(manifest_path: Path) -> List[TrainingPair]:
    """Load pairs from a JSONL manifest."""
    pairs = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pair = TrainingPair(
                pair_id=d["pair_id"],
                input_path=d["input_path"],
                target_1x_path=d["target_1x_path"],
                target_upscaled_path=d["target_upscaled_path"],
                tags=AssetTags.from_dict(d["tags"]),
                caption_short=d.get("caption_short", ""),
                caption_detailed=d.get("caption_detailed", ""),
                input_style=d.get("input_style", ""),
                quality_score=d.get("quality_score", 0.0),
                split=d.get("split", "train"),
            )
            pairs.append(pair)
    return pairs
