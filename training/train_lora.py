"""LoRA fine-tuning script for pixelization models.

Trains a LoRA adapter on a diffusion model (SD 1.5, SDXL, or Flux)
to generate pixel art from high-res images + conditioning tags.

Usage:
    python -m training.train_lora --config training_config.json
    python -m training.train_lora --manifest data/pairs/manifest.jsonl --generated-dir data/generated

The trained LoRA can be loaded with diffusers:
    pipe.load_lora_weights("training_output/final_lora")
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check that all training dependencies are available."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import diffusers
    except ImportError:
        missing.append("diffusers")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")
    try:
        import peft
    except ImportError:
        missing.append("peft")

    if missing:
        print(
            f"Missing training dependencies: {', '.join(missing)}\n"
            f"Install with:\n"
            f"  pip install torch torchvision diffusers transformers accelerate peft\n"
            f"  pip install bitsandbytes  # optional, for 8-bit Adam optimizer\n"
            f"  pip install wandb         # optional, for experiment tracking",
            file=sys.stderr,
        )
        return False
    return True


def train(config_dict: dict):
    """Run LoRA training with the given configuration.

    This is the main training loop. It:
    1. Loads the base model + tokenizer + scheduler
    2. Injects LoRA adapters
    3. Creates the training dataset
    4. Runs the training loop with validation
    5. Saves the final LoRA weights
    """
    if not check_dependencies():
        sys.exit(1)

    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    from training.config import TrainConfig
    from training.dataset import PixelArtPairDataset

    config = TrainConfig.from_dict(config_dict)

    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
    )

    alogger = get_logger(__name__)

    if accelerator.is_main_process:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
        # Save config
        config_path = Path(config.output_dir) / "train_config.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # Load base model components
    alogger.info(f"Loading base model: {config.base_model}")

    noise_scheduler = DDPMScheduler.from_pretrained(
        config.base_model, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        config.base_model, subfolder="vae",
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model, subfolder="unet",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.base_model, subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model, subfolder="text_encoder",
    )

    # Freeze base model, inject LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    alogger.info(
        f"LoRA parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            alogger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Dataset
    train_dataset = PixelArtPairDataset(
        manifest_path=config.manifest_path,
        generated_dir=config.generated_dir,
        input_size=config.input_size,
        target_size=config.target_size,
        split="train",
        augment=True,
    )
    val_dataset = PixelArtPairDataset(
        manifest_path=config.manifest_path,
        generated_dir=config.generated_dir,
        input_size=config.input_size,
        target_size=config.target_size,
        split="val",
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Calculate total steps
    if config.num_epochs:
        total_steps = len(train_loader) * config.num_epochs
    else:
        total_steps = config.max_train_steps

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare with accelerator
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler,
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    alogger.info(f"Starting training for {total_steps} steps")
    alogger.info(f"  Train samples: {len(train_dataset)}")
    alogger.info(f"  Val samples:   {len(val_dataset)}")
    alogger.info(f"  Batch size:    {config.batch_size} x {config.gradient_accumulation_steps} accum")
    alogger.info(f"  Learning rate: {config.learning_rate}")
    alogger.info(f"  LoRA rank:     {config.lora_rank}")

    unet.train()

    while global_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(unet):
                # Encode target pixel art to latents
                target_latents = vae.encode(
                    batch["target"].to(dtype=vae.dtype)
                ).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                # Build text conditioning
                # Format: "<tags> <caption>"
                prompts = []
                for tags, caption in zip(batch["tags"], batch["caption"]):
                    if config.tag_prefix and tags:
                        prompts.append(f"{tags} {caption}")
                    else:
                        prompts.append(caption)

                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)

                encoder_hidden_states = text_encoder(
                    text_inputs.input_ids
                )[0]

                # Sample noise and timesteps
                noise = torch.randn_like(target_latents)
                if config.noise_offset > 0:
                    noise += config.noise_offset * torch.randn(
                        target_latents.shape[0], target_latents.shape[1],
                        1, 1, device=target_latents.device,
                    )

                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=target_latents.device,
                ).long()

                # Add noise to target latents
                noisy_latents = noise_scheduler.add_noise(
                    target_latents, noise, timesteps,
                )

                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Loss
                if config.snr_gamma is not None:
                    # Min-SNR weighting
                    snr = _compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, config.snr_gamma * torch.ones_like(snr)],
                        dim=1,
                    ).min(dim=1)[0] / snr
                    loss = F.mse_loss(
                        noise_pred.float(), noise.float(), reduction="none"
                    )
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        config.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                # Log
                if global_step % 10 == 0 and accelerator.is_main_process:
                    alogger.info(
                        f"step {global_step}/{total_steps} | "
                        f"loss={loss.item():.4f} | "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                # Save checkpoint
                if (global_step % config.save_every_n_steps == 0
                        and accelerator.is_main_process):
                    save_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                    _save_lora(accelerator.unwrap_model(unet), save_path)
                    alogger.info(f"Saved checkpoint: {save_path}")

                if global_step >= total_steps:
                    break

    # Save final LoRA weights
    if accelerator.is_main_process:
        final_path = Path(config.output_dir) / "final_lora"
        _save_lora(accelerator.unwrap_model(unet), final_path)
        alogger.info(f"Training complete. Final LoRA saved to: {final_path}")

    accelerator.end_training()


def _compute_snr(scheduler, timesteps):
    """Compute signal-to-noise ratio for min-SNR weighting."""
    import torch
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


def _save_lora(unet, save_path: Path):
    """Save LoRA weights from a PEFT model."""
    save_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for pixel art generation",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training config JSON file",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to dataset manifest.jsonl",
    )
    parser.add_argument(
        "--generated-dir", type=str, default=None,
        help="Directory containing generated high-res images",
    )
    parser.add_argument(
        "--output-dir", type=str, default="training_output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--mixed-precision", type=str, default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and dataset stats without training",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build config
    if args.config:
        config_dict = json.loads(Path(args.config).read_text())
    else:
        config_dict = {}

    # CLI overrides
    if args.manifest:
        config_dict["manifest_path"] = args.manifest
    if args.generated_dir:
        config_dict["generated_dir"] = args.generated_dir
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.base_model:
        config_dict["base_model"] = args.base_model
    if args.learning_rate is not None:
        config_dict["learning_rate"] = args.learning_rate
    if args.lora_rank is not None:
        config_dict["lora_rank"] = args.lora_rank
    if args.max_train_steps is not None:
        config_dict["max_train_steps"] = args.max_train_steps
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.mixed_precision:
        config_dict["mixed_precision"] = args.mixed_precision

    if args.dry_run:
        from training.config import TrainConfig
        config = TrainConfig.from_dict(config_dict)
        print("Training configuration:")
        print(json.dumps(config.to_dict(), indent=2))

        if check_dependencies():
            from training.dataset import PixelArtPairDataset
            for split in ["train", "val", "test"]:
                ds = PixelArtPairDataset(
                    manifest_path=config.manifest_path,
                    generated_dir=config.generated_dir,
                    input_size=config.input_size,
                    target_size=config.target_size,
                    split=split,
                    augment=False,
                )
                print(f"  {split}: {len(ds)} pairs")
        return

    train(config_dict)


if __name__ == "__main__":
    main()
