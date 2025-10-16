"""
Dataset preparation utility for ML-assisted grid estimation.

Creates a manifest that combines human-labelled feedback with synthetic samples.
Run this script whenever new feedback is collected to refresh ``ml/data``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageFilter

from .dataset import MANIFEST_FILENAME, ManifestEntry


DEFAULT_OUTPUT_DIR = Path("ml/data")
FEEDBACK_LOG = Path("notes/feedback_log.csv")
SYNTHETIC_DIR = DEFAULT_OUTPUT_DIR / "synthetic"
RNG = random.Random()


@dataclass
class RawLabel:
    image_path: Path
    cell_size: int
    offset_x: int
    offset_y: int
    label_source: str
    weight: float


def parse_feedback(feedback_csv: Path) -> List[RawLabel]:
    """Extract positive labels from the feedback log."""
    labels: List[RawLabel] = []
    if not feedback_csv.exists():
        return labels

    with feedback_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            feedback = row.get("feedback", "").strip().lower()
            if feedback != "looks correct":
                continue

            image_path = Path(row["image"]).resolve()
            if not image_path.exists():
                continue

            cell_size = _parse_int(row.get("override_cell_size"), row.get("cell_size"))
            offset_x = _parse_int(row.get("override_offset_x"), row.get("offset_x"))
            offset_y = _parse_int(row.get("override_offset_y"), row.get("offset_y"))
            labels.append(
                RawLabel(
                    image_path=image_path,
                    cell_size=cell_size,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    label_source="feedback",
                    weight=1.0,
                )
            )
    return labels


def _parse_int(*values) -> int:
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        try:
            return int(round(float(value)))
        except ValueError:
            continue
    raise ValueError("Expected at least one numeric value")


def generate_synthetic_samples(
    output_dir: Path, count: int, seed: int
) -> List[RawLabel]:
    """Generate synthetic pixel art and return manifest entries."""
    if count <= 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    RNG.seed(seed)
    NP_RNG = np.random.default_rng(seed)

    labels: List[RawLabel] = []
    for idx in range(count):
        grid_size = RNG.choice([4, 6, 8, 12, 16, 20, 24, 28, 32])
        logical_w = RNG.randint(12, 32)
        logical_h = RNG.randint(12, 32)

        sprite = NP_RNG.integers(
            low=0, high=255, size=(logical_h, logical_w, 3), dtype=np.uint8
        )
        sprite_img = Image.fromarray(sprite, mode="RGB")
        base = sprite_img.resize(
            (logical_w * grid_size, logical_h * grid_size), resample=Image.NEAREST
        )

        offset_x = RNG.randint(0, max(0, grid_size - 1))
        offset_y = RNG.randint(0, max(0, grid_size - 1))
        canvas = Image.new(
            "RGB",
            (
                base.width + offset_x + RNG.randint(0, grid_size),
                base.height + offset_y + RNG.randint(0, grid_size),
            ),
            color=(0, 0, 0),
        )
        canvas.paste(base, (offset_x, offset_y))

        if RNG.random() < 0.6:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=RNG.uniform(0.5, 1.5)))
        if RNG.random() < 0.5:
            noise = NP_RNG.normal(0, 12, canvas.size[0] * canvas.size[1] * 3)
            noise = noise.reshape(canvas.size[1], canvas.size[0], 3)
            noisy = np.clip(
                np.array(canvas, dtype=np.int16) + noise.astype(np.int16), 0, 255
            ).astype(np.uint8)
            canvas = Image.fromarray(noisy, mode="RGB")

        out_path = output_dir / f"synthetic_{idx:05d}.png"
        canvas.save(out_path, format="PNG")

        labels.append(
            RawLabel(
                image_path=out_path.resolve(),
                cell_size=grid_size,
                offset_x=offset_x,
                offset_y=offset_y,
                label_source="synthetic",
                weight=0.4,
            )
        )
    return labels


def build_manifest(
    labels: Iterable[RawLabel], output_dir: Path, val_ratio: float, seed: int
) -> List[ManifestEntry]:
    """Write manifest.jsonl with split assignments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_list = list(labels)
    if not labels_list:
        return []

    rng = random.Random(seed)
    rng.shuffle(labels_list)

    val_count = max(1, int(len(labels_list) * val_ratio))
    if val_count >= len(labels_list):
        val_count = max(1, len(labels_list) // 5 or 1)

    manifest_entries: List[ManifestEntry] = []
    for idx, label in enumerate(labels_list):
        split = "val" if idx < val_count else "train"
        manifest_entries.append(
            ManifestEntry(
                path=label.image_path,
                cell_size=label.cell_size,
                offset_x=label.offset_x,
                offset_y=label.offset_y,
                label_source=label.label_source,
                weight=label.weight,
                split=split,
            )
        )

    manifest_path = output_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as fh:
        for entry in manifest_entries:
            fh.write(
                json.dumps(
                    {
                        "image_path": str(entry.path),
                        "cell_size": entry.cell_size,
                        "offset_x": entry.offset_x,
                        "offset_y": entry.offset_y,
                        "label_source": entry.label_source,
                        "weight": entry.weight,
                        "split": entry.split,
                    }
                )
                + "\n"
            )

    return manifest_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset manifest for grid estimation model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store manifest and synthetic assets.",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=512,
        help="Number of synthetic samples to generate (0 to skip).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback_labels = parse_feedback(FEEDBACK_LOG)
    synthetic_labels = generate_synthetic_samples(
        output_dir=args.output_dir / "synthetic",
        count=args.synthetic_count,
        seed=args.seed,
    )

    combined = feedback_labels + synthetic_labels
    if not combined:
        print(
            "No labelled data found. Add feedback via the GUI or "
            "generate synthetic samples with --synthetic-count."
        )
        return

    entries = build_manifest(
        labels=combined,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    manifest_path = args.output_dir / MANIFEST_FILENAME
    print(
        f"Manifest written to {manifest_path} "
        f"({sum(1 for e in entries if e.split == 'train')} train / "
        f"{sum(1 for e in entries if e.split == 'val')} val)"
    )


if __name__ == "__main__":
    main()
