"""
Diagnostic script comparing model predictions and heuristic estimates.

Outputs mean absolute errors for:
    * Direct model regression (single forward pass)
    * Top-k ranked suggestions (model + heuristic blend)
    * Heuristic-only estimate
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from .dataset import MAX_NORMALISATION, GridDataset, load_manifest
from .heuristics import detect_grid_from_image
from .inference import suggest_parameters
from .model import GridEstimatorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse prediction errors on the validation split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/checkpoints/grid_estimator_best.pt"),
        help="Checkpoint for the trained model.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/data/manifest.jsonl"),
        help="Dataset manifest.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Image resolution used during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of suggestions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    entries = load_manifest(args.manifest)
    val_entries = [e for e in entries if e.split == "val"]
    if not val_entries:
        raise RuntimeError("Manifest does not have validation entries.")

    dataset = GridDataset(val_entries, image_size=args.image_size, augment=False)

    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
    except (TypeError, RuntimeError, AttributeError, pickle.UnpicklingError):
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = GridEstimatorModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_errors: list[np.ndarray] = []
    suggestion_errors: list[np.ndarray] = []
    heuristic_errors: list[np.ndarray] = []

    for idx, entry in enumerate(dataset.entries):
        stacked, _, _, scale = dataset[idx]
        scale = float(scale.item())
        tensor = stacked.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0]

        pred_original = (pred * MAX_NORMALISATION) / max(scale, 1e-6)
        gt = np.array([entry.cell_size, entry.offset_x, entry.offset_y], dtype=float)
        model_errors.append(np.abs(pred_original - gt))

        suggestions = suggest_parameters(
            entry.path,
            checkpoint=args.checkpoint,
            device=str(device),
            image_size=args.image_size,
            top_k=args.top_k,
        )
        if suggestions:
            best = min(
                suggestions,
                key=lambda s: abs(s.cell_size - gt[0])
                + abs(s.offset[0] - gt[1])
                + abs(s.offset[1] - gt[2]),
            )
            suggestion_errors.append(
                np.array(
                    [
                        abs(best.cell_size - gt[0]),
                        abs(best.offset[0] - gt[1]),
                        abs(best.offset[1] - gt[2]),
                    ]
                )
            )

        heuristic = detect_grid_from_image(entry.path)
        if heuristic is not None:
            heuristic_errors.append(
                np.array(
                    [
                        abs(heuristic.cell_size - gt[0]),
                        abs(heuristic.offset_x - gt[1]),
                        abs(heuristic.offset_y - gt[2]),
                    ]
                )
            )

    def summarise(name: str, errors: list[np.ndarray]) -> None:
        if not errors:
            print(f"{name}: no data")
            return
        arr = np.stack(errors, axis=0)
        means = arr.mean(axis=0)
        medians = np.median(arr, axis=0)
        within_one = (arr <= 1).all(axis=1).mean()
        print(
            f"{name}: mean_abs=[{means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}] "
            f"median_abs=[{medians[0]:.2f}, {medians[1]:.2f}, {medians[2]:.2f}] "
            f"within_1px={within_one:.3f}"
        )

    summarise("Direct model", model_errors)
    summarise("Top-k suggestion", suggestion_errors)
    summarise("Heuristic-only", heuristic_errors)


if __name__ == "__main__":
    main()
