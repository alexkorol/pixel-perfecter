"""
Inference helpers for the grid estimation model.

Provides utilities to load trained checkpoints and produce ranked parameter
suggestions for a given input image.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .dataset import MAX_NORMALISATION
from .heuristics import detect_grid_from_image
from .model import GridEstimatorModel


DEFAULT_CHECKPOINT = Path("ml/checkpoints/grid_estimator_best.pt")


@dataclass
class MLSuggestion:
    cell_size: int
    offset: Tuple[int, int]
    confidence: float


def _preprocess(image_path: Path, image_size: int) -> tuple[torch.Tensor, float]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        scale = image_size / max(orig_w, orig_h)
        img = ImageOps.pad(
            img,
            (image_size, image_size),
            method=Image.BILINEAR,
            color=(0, 0, 0),
        )
        rgb = np.array(img, dtype=np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    edge_tensor = torch.from_numpy(edges).unsqueeze(0).float() / 255.0
    tensor = torch.cat([rgb_tensor, edge_tensor], dim=0).unsqueeze(0)
    return tensor, float(scale)


@lru_cache(maxsize=2)
def _load_model(checkpoint_path: Path, device: str) -> GridEstimatorModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train the model with ml.grid_estimator first."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError, AttributeError, pickle.UnpicklingError):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GridEstimatorModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def suggest_parameters(
    image_path: str | Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    image_size: int = 160,
    top_k: int = 3,
) -> List[MLSuggestion]:
    """
    Generate ranked grid parameter suggestions for ``image_path``.

    Returns a list of ``MLSuggestion`` sorted by confidence (descending).
    """
    image_path = Path(image_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = _load_model(checkpoint, device)
    except FileNotFoundError:
        return []

    tensor, scale = _preprocess(image_path, image_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        preds = model(tensor).cpu().numpy()[0]

    raw_values = preds * MAX_NORMALISATION
    scaled_cell, scaled_off_x, scaled_off_y = raw_values.astype(float)

    scale = max(scale, 1e-6)
    cell_pred = scaled_cell / scale
    off_x_pred = scaled_off_x / scale
    off_y_pred = scaled_off_y / scale

    base_cell = max(2, int(round(cell_pred)))
    candidate_cells = _candidate_cell_sizes(base_cell)
    offset_candidates = _candidate_offsets(off_x_pred, off_y_pred)

    ranked: Dict[Tuple[int, int, int], float] = {}
    heuristic = detect_grid_from_image(image_path)
    for cell_rank, cell_size in enumerate(candidate_cells):
        penalty = 0.05 * cell_rank
        for offset in offset_candidates:
            clamped_offset = _clamp_offset(offset, cell_size)
            score = _confidence_score(
                cell_pred, cell_size, off_x_pred, off_y_pred, clamped_offset
            )
            ranked[(cell_size, clamped_offset[0], clamped_offset[1])] = max(
                ranked.get((cell_size, clamped_offset[0], clamped_offset[1]), 0.0),
                max(0.05, score - penalty),
            )

    if heuristic is not None:
        h_offset = _clamp_offset((heuristic.offset_x, heuristic.offset_y), heuristic.cell_size)
        ranked[(heuristic.cell_size, h_offset[0], h_offset[1])] = max(
            ranked.get((heuristic.cell_size, h_offset[0], h_offset[1]), 0.0),
            0.35,
        )

    suggestions = [
        MLSuggestion(
            cell_size=cell,
            offset=(off_x, off_y),
            confidence=float(conf),
        )
        for (cell, off_x, off_y), conf in ranked.items()
    ]
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions[:top_k]


def _clamp_offset(offset: Tuple[int, int], cell_size: int) -> Tuple[int, int]:
    max_val = max(1, cell_size - 1)
    return (
        int(np.clip(offset[0], 0, max_val)),
        int(np.clip(offset[1], 0, max_val)),
    )


def _confidence_score(
    cell_pred: float,
    cell_choice: int,
    off_x_pred: float,
    off_y_pred: float,
    offset_choice: Tuple[int, int],
) -> float:
    cell_error = abs(cell_pred - float(cell_choice))
    offset_error = (
        abs(off_x_pred - float(offset_choice[0]))
        + abs(off_y_pred - float(offset_choice[1]))
    )
    offset_norm = offset_error / max(float(cell_choice), 1.0)
    penalty = 0.6 * cell_error + 0.4 * offset_norm
    score = math.exp(-penalty)
    return float(max(0.05, min(score, 0.99)))


def _candidate_cell_sizes(base_cell: int) -> List[int]:
    candidates: List[int] = []
    for delta in (0, -1, 1, -2, 2):
        value = max(2, base_cell + delta)
        if value not in candidates:
            candidates.append(value)
    return candidates


def _candidate_offsets(off_x_pred: float, off_y_pred: float) -> List[Tuple[int, int]]:
    x_candidates = _axis_candidates(off_x_pred)
    y_candidates = _axis_candidates(off_y_pred)

    combined: Dict[Tuple[int, int], float] = {}
    for x in x_candidates:
        for y in y_candidates:
            combined[(x, y)] = abs(off_x_pred - x) + abs(off_y_pred - y)

    sorted_items = sorted(combined.items(), key=lambda item: item[1])
    return [coords for coords, _ in sorted_items]


def _axis_candidates(value: float) -> List[int]:
    floor_val = int(np.floor(value))
    ceil_val = int(np.ceil(value))
    round_val = int(round(value))

    candidates = {floor_val, ceil_val, round_val}
    frac = value - floor_val

    if frac < 0.25:
        candidates.add(round_val - 1)
    if frac > 0.75:
        candidates.add(round_val + 1)

    return sorted(candidates)
