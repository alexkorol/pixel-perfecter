"""
Smoke tests for the batch CLI.

The goal is to exercise the full reconstruction pipeline on a tiny synthetic
sprite so regressions in wiring or filesystem layout are caught early.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from pixel_perfecter.cli import main as cli_main


def _save_checkerboard(path: Path, size: int = 32, block: int = 4) -> None:
    """Create a simple RGBA checkerboard sprite for testing."""
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    colors = [
        (32, 48, 112, 255),
        (240, 200, 96, 255),
        (20, 20, 24, 255),
        (220, 80, 92, 255),
    ]
    for y in range(size):
        for x in range(size):
            idx = ((x // block) + (y // block)) % len(colors)
            pixels[y, x] = colors[idx]
    image = Image.fromarray(pixels, mode="RGBA")
    image.save(path)


def test_cli_smoke(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    source = input_dir / "sample.png"
    _save_checkerboard(source)

    cli_main(
        [
            str(source),
            "--output-dir",
            str(output_dir),
        ]
    )

    reconstructed = output_dir / "sample_reconstructed.png"
    overlay = output_dir / "sample_validation.png"
    metrics_path = output_dir / "metrics.csv"

    assert reconstructed.exists(), "CLI did not write reconstructed sprite"
    assert overlay.exists(), "CLI did not emit diagnostic overlay"
    assert metrics_path.exists(), "CLI did not persist metrics CSV"

    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "metrics.csv is empty"
    assert rows[0]["image"] == "sample.png"
    assert rows[0]["output_path"].endswith("sample_reconstructed.png")
