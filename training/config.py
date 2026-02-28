"""Training configuration for pixelization models.

Supports multiple training approaches:
  - lora:       LoRA fine-tuning of Stable Diffusion / SDXL / Flux
  - controlnet: ControlNet training for image-to-pixel-art
  - pix2pix:    InstructPix2Pix-style training
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class TrainingApproach(str, Enum):
    LORA = "lora"
    CONTROLNET = "controlnet"
    PIX2PIX = "pix2pix"


class BaseModel(str, Enum):
    SD15 = "runwayml/stable-diffusion-v1-5"
    SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
    FLUX_DEV = "black-forest-labs/FLUX.1-dev"
    FLUX_SCHNELL = "black-forest-labs/FLUX.1-schnell"


@dataclass
class TrainConfig:
    """Full training configuration."""

    # --- Approach ---
    approach: TrainingApproach = TrainingApproach.LORA
    base_model: str = BaseModel.SDXL.value

    # --- Data ---
    manifest_path: str = "data/pairs/manifest.jsonl"
    generated_dir: str = "data/generated"
    input_size: int = 512       # resize inputs to this
    target_size: int = 512      # resize targets to this

    # --- Training hyperparameters ---
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    max_train_steps: int = 5000
    num_epochs: Optional[int] = None  # if set, overrides max_train_steps

    # --- LoRA-specific ---
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
    ])

    # --- Regularization ---
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    noise_offset: float = 0.0   # noise offset for better dark/bright images
    snr_gamma: Optional[float] = 5.0  # min-SNR weighting

    # --- Hardware ---
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    num_workers: int = 4

    # --- Output ---
    output_dir: str = "training_output"
    logging_dir: str = "training_output/logs"
    save_every_n_steps: int = 500
    validate_every_n_steps: int = 250
    sample_every_n_steps: int = 250
    num_validation_samples: int = 4

    # --- Conditioning ---
    # Tag string is prepended to caption for text conditioning
    tag_prefix: bool = True  # if True, conditions on "grid:32px outline:black ... <caption>"

    # --- Resume ---
    resume_from: Optional[str] = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        if "approach" in d and isinstance(d["approach"], str):
            d["approach"] = TrainingApproach(d["approach"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_accelerate_args(self) -> List[str]:
        """Generate accelerate launch arguments."""
        args = [
            f"--mixed_precision={self.mixed_precision}",
        ]
        if self.gradient_checkpointing:
            args.append("--gradient_checkpointing")
        return args
