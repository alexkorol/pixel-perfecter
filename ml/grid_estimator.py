"""
Training script for ML-assisted grid parameter estimation.

Relies on datasets prepared via ``ml.prepare_dataset`` and supports GPU training,
TensorBoard logging, and validation metrics (MAE / accuracy thresholds).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import MAX_NORMALISATION, GridDataset, load_manifest, split_entries
from .model import GridEstimatorModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


DEFAULT_MANIFEST = Path("ml/data/manifest.jsonl")
CHECKPOINT_DIR = Path("ml/checkpoints")


@dataclass
class TrainConfig:
    manifest_path: Path = DEFAULT_MANIFEST
    image_size: int = 160
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 20
    val_ratio: float = 0.2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Optional[Path] = Path("ml/runs")
    amp: bool = False


def create_dataloaders(cfg: TrainConfig):
    entries = load_manifest(cfg.manifest_path)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    if not train_entries or not val_entries:
        train_entries, val_entries = split_entries(entries, cfg.val_ratio, cfg.seed)
        for entry in train_entries:
            entry.split = "train"
        for entry in val_entries:
            entry.split = "val"

    if not train_entries:
        raise RuntimeError("No training samples available. Run prepare_dataset first.")

    train_dataset = GridDataset(
        train_entries, image_size=cfg.image_size, augment=True
    )
    val_dataset = GridDataset(val_entries, image_size=cfg.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=cfg.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.device.startswith("cuda"),
    )
    return train_loader, val_loader


def weighted_mse_loss(
    preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    diff = preds - targets
    mse = diff.pow(2)
    mae = diff.abs()
    blended = 0.7 * mse + 0.3 * mae
    per_sample = blended.mean(dim=1)
    weight_sum = torch.clamp(weights.sum(), min=1e-6)
    return (per_sample * weights).sum() / weight_sum


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    diff_scaled = torch.abs(preds - targets) * MAX_NORMALISATION
    scales = scales.view(-1, 1).clamp_min(1e-6)
    diff_original = diff_scaled / scales
    mae = diff_original.mean(dim=0)
    within_one = (diff_original <= 1.0).all(dim=1).float().mean()
    return mae, within_one


def train(cfg: TrainConfig) -> None:
    train_loader, val_loader = create_dataloaders(cfg)
    device = torch.device(cfg.device)

    model = GridEstimatorModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - 1)
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = GradScaler(enabled=True) if use_amp else None

    writer = None
    if SummaryWriter is not None and cfg.log_dir:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(cfg.log_dir))

    best_val_mae: Optional[float] = None

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for images, targets, weights, scales in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            scales = scales.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with autocast(device_type="cuda", enabled=True):
                    preds = model(images)
                    loss = weighted_mse_loss(preds, targets, weights)
            else:
                preds = model(images)
                loss = weighted_mse_loss(preds, targets, weights)

            if not torch.isfinite(loss):
                print("Skipping step due to non-finite loss", float(loss))
                continue

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            sample_count += images.size(0)

        scheduler.step()
        avg_loss = running_loss / max(1, sample_count)

        val_loss, val_mae, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} "
            f"- train_loss: {avg_loss:.6f} "
            f"- val_loss: {val_loss:.6f} "
            f"- val_mae(px): {[round(x, 3) for x in val_mae.tolist()]} "
            f"- val_acc(|delta|<=1px): {val_acc:.3f}"
        )

        if writer:
            writer.add_scalar("loss/train", avg_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/val_acc_within_1px", val_acc, epoch)
            writer.add_scalar("metrics/val_mae_cell", val_mae[0], epoch)
            writer.add_scalar("metrics/val_mae_offset_x", val_mae[1], epoch)
            writer.add_scalar("metrics/val_mae_offset_y", val_mae[2], epoch)

        mean_mae = float(val_mae.mean().item())
        if best_val_mae is None or mean_mae < best_val_mae:
            best_val_mae = mean_mae
            save_checkpoint(model, cfg, suffix="best")

    save_checkpoint(model, cfg, suffix="last")
    if writer:
        writer.close()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    total_mae = torch.zeros(3, device=device)
    total_acc = 0.0
    total_samples = 0

    for images, targets, weights, scales in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        scales = scales.to(device, non_blocking=True)

        preds = model(images)
        loss = weighted_mse_loss(preds, targets, weights)

        mae, acc = compute_metrics(preds, targets, scales)

        batch_weight = weights.sum().item()
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight
        total_mae += mae * images.size(0)
        total_acc += acc.item() * images.size(0)
        total_samples += images.size(0)

    mean_loss = total_loss / max(total_weight, 1e-6)
    mean_mae = (total_mae / max(total_samples, 1)).cpu()
    mean_acc = total_acc / max(total_samples, 1)
    return mean_loss, mean_mae, mean_acc


def save_checkpoint(model: nn.Module, cfg: TrainConfig, suffix: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"grid_estimator_{suffix}.pt"
    config_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in cfg.__dict__.items()
    }
    torch.save({"model_state_dict": model.state_dict(), "config": config_dict}, path)
    print(f"Checkpoint saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grid estimation model.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=TrainConfig.log_dir,
        help="TensorBoard log directory (omit to disable).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP mixed precision (disabled by default for stability).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        manifest_path=args.manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        log_dir=args.log_dir,
        amp=args.amp,
    )
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
