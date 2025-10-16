"""
Utility to evaluate a saved grid estimator checkpoint on a manifest split.

Usage:
    python -m ml.evaluate_checkpoint --checkpoint ml/checkpoints/grid_estimator_best.pt
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from .grid_estimator import TrainConfig, create_dataloaders, evaluate
from .model import GridEstimatorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained grid estimator checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/checkpoints/grid_estimator_best.pt"),
        help="Path to the checkpoint produced by ml.grid_estimator.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/data/manifest.jsonl"),
        help="Manifest to evaluate against.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for evaluation dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Image resolution used during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = TrainConfig(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=str(device),
    )

    _, val_loader = create_dataloaders(cfg)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
    except (TypeError, RuntimeError, AttributeError, pickle.UnpicklingError):
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = GridEstimatorModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, mae, acc = evaluate(model, val_loader, device)
    metrics = {
        "val_loss": float(loss),
        "mae_cell_px": float(mae[0].item()),
        "mae_offset_x_px": float(mae[1].item()),
        "mae_offset_y_px": float(mae[2].item()),
        "accuracy_within_1px": float(acc),
    }

    print(
        " | ".join(
            f"{key}: {round(value, 3) if isinstance(value, float) else value}"
            for key, value in metrics.items()
        )
    )


if __name__ == "__main__":
    main()
