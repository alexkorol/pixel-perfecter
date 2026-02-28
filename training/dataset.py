"""PyTorch dataset for pixelization model training.

Loads (high-res input, pixel art target) pairs from the packaged
dataset and applies training-time augmentations.

The model learns: high-res image + conditioning tags → pixel art sprite.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False


class PixelArtPairDataset:
    """Dataset of (high-res input, pixel art target) pairs.

    Loads from the packaged dataset manifest.

    Each sample returns:
        input_image:  (3, H, W) tensor — high-res generated image
        target_image: (3, H, W) tensor — pixel art (NN-upscaled to match)
        tags:         str — conditioning tag string
        caption:      str — short description
    """

    def __init__(
        self,
        manifest_path: str,
        generated_dir: str,
        input_size: int = 512,
        target_size: int = 512,
        split: str = "train",
        augment: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch torchvision")

        self.manifest_path = Path(manifest_path)
        self.generated_dir = Path(generated_dir)
        self.input_size = input_size
        self.target_size = target_size
        self.split = split
        self.augment = augment

        # Load manifest entries for this split
        self.entries = []
        self._load_manifest()

        # Transforms
        self.input_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Pixel art targets use NEAREST interpolation to preserve hard edges
        self.target_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        if augment and split == "train":
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05,
                    saturation=0.05, hue=0.02,
                ),
            ])
        else:
            self.augmentation = None

        logger.info(
            "Loaded %d pairs for split=%s from %s",
            len(self.entries), split, manifest_path,
        )

    def _load_manifest(self):
        """Load and filter manifest entries."""
        if not self.manifest_path.exists():
            logger.warning("Manifest not found: %s", self.manifest_path)
            return

        with open(self.manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("split", "train") != self.split:
                    continue

                # Verify files exist
                sprite_path = entry.get("sprite_path", "")
                if not sprite_path or not Path(sprite_path).exists():
                    continue

                # Find a matching generated image
                stem = Path(sprite_path).stem
                gen_dir = self.generated_dir / stem
                gen_images = list(gen_dir.glob("*.png")) if gen_dir.exists() else []

                if not gen_images:
                    # Also check for flat naming: <stem>_<style>.png
                    gen_images = list(self.generated_dir.glob(f"{stem}_*.png"))

                for gen_path in gen_images:
                    self.entries.append({
                        "input_path": str(gen_path),
                        "target_path": sprite_path,
                        "target_upscaled_path": entry.get("sprite_upscaled_path", ""),
                        "tags": entry.get("tags", {}).get("tag_string", ""),
                        "caption": entry.get("caption_short", ""),
                    })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]

        # Load input (high-res generated image)
        input_img = Image.open(entry["input_path"]).convert("RGB")

        # Load target (pixel art — prefer upscaled version for matching size)
        target_path = entry.get("target_upscaled_path") or entry["target_path"]
        if not Path(target_path).exists():
            target_path = entry["target_path"]
        target_img = Image.open(target_path).convert("RGB")

        # Augment both images identically
        if self.augmentation is not None:
            # Seed random state for synchronized augmentation
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.augmentation(input_img)
            torch.manual_seed(seed)
            target_img = self.augmentation(target_img)

        # Apply transforms
        input_tensor = self.input_transform(input_img)
        target_tensor = self.target_transform(target_img)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "tags": entry["tags"],
            "caption": entry["caption"],
        }


def create_dataloaders(
    manifest_path: str,
    generated_dir: str,
    batch_size: int = 4,
    input_size: int = 512,
    target_size: int = 512,
    num_workers: int = 4,
) -> Tuple:
    """Create train/val/test dataloaders.

    Returns (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    splits = {}
    for split in ["train", "val", "test"]:
        ds = PixelArtPairDataset(
            manifest_path=manifest_path,
            generated_dir=generated_dir,
            input_size=input_size,
            target_size=target_size,
            split=split,
            augment=(split == "train"),
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        splits[split] = loader

    return splits["train"], splits["val"], splits["test"]
