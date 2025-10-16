"""
Heuristic grid detection inspired by Astropulse/pixeldetector (MIT License).

The original project applies gradient peak spacing and per-tile color clustering
to recover pixel-art grids. This module re-implements the core ideas with
modifications suited for our pipeline while preserving attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.signal import find_peaks


@dataclass
class HeuristicEstimate:
    cell_size: int
    offset_x: int
    offset_y: int
    spacing_x: np.ndarray
    spacing_y: np.ndarray


def rgb_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _peak_spacing(values: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    # Sum gradients along the requested axis to emphasise edges.
    diff = np.diff(values, axis=axis)
    diff_sq = np.square(diff).sum(axis=2)
    grad = np.sqrt(diff_sq)
    summed = grad.sum(axis=axis ^ 1)
    peaks, _ = find_peaks(summed, distance=1, height=0.0)
    return summed, peaks


def _spacing_stats(peaks: np.ndarray) -> Optional[tuple[float, float]]:
    if len(peaks) < 2:
        return None
    spacing = np.diff(peaks)
    if spacing.size == 0 or float(np.median(spacing)) <= 0.0:
        return None
    return float(np.median(spacing)), float(peaks[0])


def detect_grid_from_image(path: Path) -> Optional[HeuristicEstimate]:
    rgb = rgb_from_path(path)
    if rgb.size == 0:
        return None

    horizontal_sums, h_peaks = _peak_spacing(rgb, axis=1)
    vertical_sums, v_peaks = _peak_spacing(rgb, axis=0)

    h_stats = _spacing_stats(h_peaks)
    v_stats = _spacing_stats(v_peaks)

    if h_stats is None or v_stats is None:
        return None

    cell_size = int(round(max(1.0, (h_stats[0] + v_stats[0]) / 2.0)))
    offset_x = int(round(h_stats[1] % max(1.0, h_stats[0])))
    offset_y = int(round(v_stats[1] % max(1.0, v_stats[0])))

    return HeuristicEstimate(
        cell_size=cell_size,
        offset_x=max(0, offset_x),
        offset_y=max(0, offset_y),
        spacing_x=np.diff(h_peaks) if len(h_peaks) > 1 else np.array([]),
        spacing_y=np.diff(v_peaks) if len(v_peaks) > 1 else np.array([]),
    )

