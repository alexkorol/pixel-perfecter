"""Unified CLI for the pixel art dataset pipeline.

Usage:
    python -m pipeline.cli extract           <input> -o <output>  [--strategy auto|grid|contours|components]
    python -m pipeline.cli clean             <input> -o <output>  [--max-core-diff 8.0]
    python -m pipeline.cli tag               <input>              [--source-set dcss] [--grid-size 32]
    python -m pipeline.cli caption           <input>              [--backend claude|openai|local|manual]
    python -m pipeline.cli prompts           <manifest> -o <dir>  [--styles photo,painting,concept]
    python -m pipeline.cli generate          <manifest> -o <dir>  [--backend comfyui|a1111|openai|replicate]
    python -m pipeline.cli package           <manifest> -o <dir>  [--val-ratio 0.1] [--test-ratio 0.1]
    python -m pipeline.cli train             --manifest <path> --generated-dir <dir>  [--config config.json]
    python -m pipeline.cli crawl-descriptions -o <dir>  [--fetch | --tiles-dir <path> --crawl-dir <path>]
    python -m pipeline.cli run               <input> -o <output>  [--source-set dcss]  # full pipeline

Each subcommand corresponds to a pipeline stage:
  extract           — Split sprite sheets/showcases into individual sprites
  clean             — Pixel-perfect reconstruction of extracted sprites
  tag               — Auto-detect metadata tags for cleaned sprites
  caption           — Describe sprites using a VLM for accurate prompt generation
  prompts           — Generate unpixelization prompts for pair creation
  generate          — Generate high-res counterpart images from prompts
  package           — Assemble final training dataset from completed pairs
  train             — Train a pixelization model (LoRA) on the packaged dataset
  crawl-descriptions — Datamine DCSS tiles + descriptions → curation gallery
  run               — Run extract → clean → tag → caption → prompts in sequence
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("pipeline")


def _setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _gather_images(inputs: List[str], recursive: bool = False) -> List[Path]:
    """Collect image paths from file/directory arguments."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p)
        elif p.is_dir():
            if recursive:
                for ext in extensions:
                    paths.extend(sorted(p.rglob(f"*{ext}")))
            else:
                for ext in extensions:
                    paths.extend(sorted(p.glob(f"*{ext}")))
    return paths


# ---- Subcommand: extract ----

def cmd_extract(args):
    from pipeline.sheet_splitter import (
        extract_sprites, save_sprites, deduplicate_sprites,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _gather_images(args.inputs, recursive=args.recursive)
    if not image_paths:
        logger.error("No images found in %s", args.inputs)
        return 1

    total_extracted = 0
    total_deduped = 0

    for img_path in image_paths:
        logger.info("Extracting from %s (strategy=%s)", img_path.name, args.strategy)

        cell_size = None
        if args.cell_size:
            parts = args.cell_size.split("x")
            cell_size = (int(parts[0]), int(parts[1]) if len(parts) > 1 else int(parts[0]))

        sprites = extract_sprites(
            str(img_path),
            strategy=args.strategy,
            cell_size=cell_size,
            min_size=args.min_size,
        )

        before = len(sprites)
        if args.deduplicate:
            sprites = deduplicate_sprites(sprites)
        after = len(sprites)

        sprite_dir = output_dir / img_path.stem
        saved = save_sprites(sprites, sprite_dir, prefix=img_path.stem)

        total_extracted += before
        total_deduped += (before - after)

        logger.info(
            "  → %d sprites extracted, %d duplicates removed, %d saved",
            before, before - after, len(saved),
        )

    logger.info(
        "Total: %d extracted, %d duplicates removed, %d unique",
        total_extracted, total_deduped, total_extracted - total_deduped,
    )
    return 0


# ---- Subcommand: clean ----

def cmd_clean(args):
    from pipeline.cleaner import bulk_clean

    input_dir = Path(args.inputs[0])
    if not input_dir.is_dir():
        # single file or list of files — wrap in a temp dir logic
        logger.error("Clean expects a directory of sprite images")
        return 1

    output_dir = Path(args.output)
    results = bulk_clean(
        input_dir,
        output_dir,
        use_hough=not args.no_hough,
        upscale_target=args.upscale_target,
        max_core_diff=args.max_core_diff,
    )

    logger.info("Cleaned %d sprites → %s", len(results), output_dir)
    return 0


# ---- Subcommand: tag ----

def cmd_tag(args):
    from pipeline.tagger import batch_tag

    input_dir = args.inputs[0]
    results = batch_tag(
        input_dir,
        grid_size=args.grid_size,
        source_set=args.source_set or "",
    )

    # output as JSONL
    output_path = Path(args.output) if args.output else Path(input_dir) / "tags.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for filename, tags in results:
            entry = {"filename": filename, **tags.to_dict()}
            f.write(json.dumps(entry) + "\n")

    logger.info("Tagged %d sprites → %s", len(results), output_path)

    # also print summary
    from collections import Counter
    outlines = Counter(t.outline.value for _, t in results)
    shadings = Counter(t.shading.value for _, t in results)
    palette_sizes = [t.palette_count for _, t in results]

    logger.info("Outline types: %s", dict(outlines))
    logger.info("Shading styles: %s", dict(shadings))
    if palette_sizes:
        logger.info(
            "Palette sizes: min=%d, max=%d, median=%d",
            min(palette_sizes), max(palette_sizes),
            sorted(palette_sizes)[len(palette_sizes) // 2],
        )

    return 0


# ---- Subcommand: prompts ----

def cmd_prompts(args):
    from pipeline.pair_generator import batch_generate_prompts
    from pipeline.config import InputStyle

    manifest_path = Path(args.inputs[0])
    output_dir = Path(args.output)

    styles = None
    if args.styles:
        style_names = [s.strip() for s in args.styles.split(",")]
        styles = [InputStyle(s) for s in style_names]

    count = batch_generate_prompts(manifest_path, output_dir, styles)
    logger.info("Generated prompts for %d sprites", count)
    return 0


# ---- Subcommand: caption ----

def cmd_caption(args):
    from pipeline.captioner import batch_caption

    input_dir = Path(args.inputs[0])
    if not input_dir.is_dir():
        logger.error("Caption expects a directory of sprite images")
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else input_dir / "captions"

    results = batch_caption(
        input_dir,
        backend=args.backend,
        model=args.model,
        cache_dir=cache_dir,
        rate_limit=args.rate_limit,
    )

    logger.info("Captioned %d sprites → %s", len(results), cache_dir)
    for name, caption in results[:5]:
        logger.info("  %s: %s", name, caption.short_description)
    if len(results) > 5:
        logger.info("  ... and %d more", len(results) - 5)

    return 0


# ---- Subcommand: generate ----

def cmd_generate(args):
    from pipeline.generator import generate_pairs_from_manifest, GenerationConfig
    from pipeline.config import InputStyle

    manifest_path = Path(args.inputs[0])
    output_dir = Path(args.output)

    config = GenerationConfig(
        width=args.width,
        height=args.height,
        num_images=args.num_images,
        guidance_scale=args.guidance_scale,
        num_steps=args.steps,
        seed=args.seed,
        model=args.model or "",
    )

    styles = None
    if args.styles:
        style_names = [s.strip() for s in args.styles.split(",")]
        styles = [InputStyle(s) for s in style_names]

    count = generate_pairs_from_manifest(
        manifest_path,
        output_dir,
        config=config,
        backend=args.backend,
        styles=styles,
        rate_limit=args.rate_limit,
        skip_existing=not args.regenerate,
    )

    logger.info("Generated %d images → %s", count, output_dir)
    return 0


# ---- Subcommand: train ----

def cmd_train(args):
    from training.train_lora import main as train_main

    # Build argv for the training script
    train_argv = []
    if args.config:
        train_argv.extend(["--config", args.config])
    if args.manifest:
        train_argv.extend(["--manifest", args.manifest])
    if args.generated_dir:
        train_argv.extend(["--generated-dir", args.generated_dir])
    if args.train_output:
        train_argv.extend(["--output-dir", args.train_output])
    if args.base_model:
        train_argv.extend(["--base-model", args.base_model])
    if args.learning_rate is not None:
        train_argv.extend(["--learning-rate", str(args.learning_rate)])
    if args.lora_rank is not None:
        train_argv.extend(["--lora-rank", str(args.lora_rank)])
    if args.max_train_steps is not None:
        train_argv.extend(["--max-train-steps", str(args.max_train_steps)])
    if args.dry_run:
        train_argv.append("--dry-run")

    train_main(train_argv)
    return 0


# ---- Subcommand: package ----

def cmd_package(args):
    from pipeline.packager import load_manifest, package_dataset

    manifest_path = Path(args.inputs[0])
    output_dir = Path(args.output)

    pairs = load_manifest(manifest_path)
    if not pairs:
        logger.error("No pairs found in manifest")
        return 1

    stats = package_dataset(
        pairs,
        output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    logger.info("Dataset stats: %s", json.dumps(stats.to_dict(), indent=2))
    return 0


# ---- Subcommand: crawl-descriptions ----

def cmd_crawl_descriptions(args):
    from pipeline.crawl_descriptions import main as crawl_main

    crawl_argv = ["-o", args.output]
    if args.tiles_dir:
        crawl_argv.extend(["--tiles-dir", args.tiles_dir])
    if args.crawl_dir:
        crawl_argv.extend(["--crawl-dir", args.crawl_dir])
    if args.fetch:
        crawl_argv.append("--fetch")
    if args.categories:
        crawl_argv.extend(["--categories"] + args.categories)
    if args.no_embed:
        crawl_argv.append("--no-embed")
    if args.selections:
        crawl_argv.extend(["--selections", args.selections])

    return crawl_main(crawl_argv)


# ---- Subcommand: run (full pipeline) ----

def cmd_run(args):
    """Run the full pipeline: extract → clean → tag → caption → prompts."""
    from pipeline.sheet_splitter import extract_sprites, save_sprites, deduplicate_sprites
    from pipeline.cleaner import clean_sprite
    from pipeline.tagger import auto_tag
    from pipeline.pair_generator import (
        generate_prompts, save_prompts, caption_placeholder,
    )
    from pipeline.config import PipelineConfig, InputStyle, AssetTags

    # Try to import VLM captioner (optional)
    captioner_available = False
    captioner_backend = getattr(args, "captioner_backend", None)
    if captioner_backend and captioner_backend != "none":
        try:
            from pipeline.captioner import caption_sprite
            captioner_available = True
            logger.info("VLM captioning enabled (backend=%s)", captioner_backend)
        except ImportError as e:
            logger.warning("VLM captioning not available: %s", e)

    config = PipelineConfig()
    if args.output:
        base = Path(args.output)
        config.sprites_dir = base / "sprites"
        config.cleaned_dir = base / "cleaned"
        config.prompts_dir = base / "prompts"
        config.manifest_path = base / "manifest.jsonl"
    config.ensure_dirs()

    caption_cache_dir = config.cleaned_dir.parent / "captions"
    caption_cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _gather_images(args.inputs, recursive=args.recursive)
    if not image_paths:
        logger.error("No images found")
        return 1

    styles = None
    if args.styles:
        style_names = [s.strip() for s in args.styles.split(",")]
        styles = [InputStyle(s) for s in style_names]
    else:
        styles = [InputStyle.PHOTOREALISTIC, InputStyle.DIGITAL_PAINTING,
                  InputStyle.CONCEPT_ART]

    source_set = args.source_set or ""
    manifest_entries = []
    stats = {"extracted": 0, "cleaned": 0, "tagged": 0, "captioned": 0, "prompts": 0}

    for img_path in image_paths:
        logger.info("=" * 60)
        logger.info("Processing: %s", img_path.name)

        # --- Stage 1: Extract sprites ---
        sprites = extract_sprites(str(img_path), strategy=args.strategy)
        sprites = deduplicate_sprites(sprites)
        stats["extracted"] += len(sprites)

        if not sprites:
            logger.info("  No sprites extracted, treating as single sprite")
            pil = Image.open(img_path)
            if pil.mode == "RGBA":
                arr = np.array(pil)
            elif pil.mode == "P":
                arr = np.array(pil.convert("RGBA"))
            else:
                arr = np.array(pil.convert("RGB"))

            from pipeline.sheet_splitter import ExtractedSprite, _perceptual_hash
            sprites = [ExtractedSprite(
                image=arr,
                bbox=(0, 0, arr.shape[1], arr.shape[0]),
                source_path=str(img_path),
                index=0,
                sprite_hash=_perceptual_hash(arr),
            )]
            stats["extracted"] += 1

        saved_sprites = save_sprites(
            sprites, config.sprites_dir / img_path.stem, prefix=img_path.stem,
        )
        logger.info("  Extracted %d sprites", len(sprites))

        # --- Stage 2: Clean each sprite ---
        for sprite, sprite_path in zip(sprites, saved_sprites):
            asset = clean_sprite(
                sprite.image,
                source_path=str(sprite_path),
                use_hough=not args.no_hough,
            )

            if asset is None:
                logger.info("  Skipped %s (cleaning failed)", sprite_path.name)
                continue

            stats["cleaned"] += 1

            # save cleaned versions
            stem = sprite_path.stem
            clean_1x = config.cleaned_dir / f"{stem}_1x.png"
            clean_up = config.cleaned_dir / f"{stem}_up.png"

            channels = asset.image_1x.shape[2] if len(asset.image_1x.shape) == 3 else 1
            pil_mode = "RGBA" if channels == 4 else "RGB"
            Image.fromarray(asset.image_1x, pil_mode).save(clean_1x)
            Image.fromarray(asset.image_upscaled, pil_mode).save(clean_up)

            # --- Stage 3: Tag ---
            tags = auto_tag(
                asset.image_1x,
                grid_size=asset.cell_size,
                source_set=source_set,
                source_name=sprite_path.name,
            )
            stats["tagged"] += 1

            logger.info(
                "  Cleaned %s: %s",
                stem, tags.to_tag_string(),
            )

            # --- Stage 3b: Caption (VLM or placeholder) ---
            if captioner_available:
                try:
                    caption = caption_sprite(
                        asset.image_1x,
                        source_path=str(clean_1x),
                        backend=captioner_backend,
                        cache_dir=caption_cache_dir,
                    )
                    stats["captioned"] += 1
                    logger.info("  Captioned: %s", caption.short_description)
                except Exception as e:
                    logger.warning("  Caption failed, using placeholder: %s", e)
                    caption = caption_placeholder(stem)
            else:
                caption = caption_placeholder(stem)

            # --- Stage 4: Generate prompts ---
            prompts = generate_prompts(
                str(clean_1x), tags, caption, styles,
            )
            prompt_dir = config.prompts_dir / stem
            save_prompts(prompts, prompt_dir, stem)
            stats["prompts"] += len(prompts)

            # --- Add to manifest ---
            manifest_entries.append({
                "sprite_path": str(clean_1x),
                "sprite_upscaled_path": str(clean_up),
                "source_image": str(img_path),
                "source_sprite_index": sprite.index,
                "tags": tags.to_dict(),
                "caption_short": caption.short_description,
                "caption_detailed": caption.detailed_description,
                "caption_source": caption.source,
                "cell_size": asset.cell_size,
                "metrics": {
                    k: v for k, v in asset.metrics.items()
                    if isinstance(v, (int, float, str, bool))
                },
            })

    # Write manifest
    with open(config.manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info("=" * 60)
    logger.info("Pipeline complete:")
    logger.info("  Extracted:  %d sprites", stats["extracted"])
    logger.info("  Cleaned:    %d assets", stats["cleaned"])
    logger.info("  Tagged:     %d assets", stats["tagged"])
    logger.info("  Captioned:  %d assets (VLM)", stats["captioned"])
    logger.info("  Prompts:    %d generated", stats["prompts"])
    logger.info("  Manifest:   %s", config.manifest_path)

    return 0


# ---- Argument parser ----

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Pixel art dataset pipeline for pixelization model training.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract --
    p_extract = sub.add_parser("extract", help="Extract sprites from sheets/showcases")
    p_extract.add_argument("inputs", nargs="+", help="Image files or directories")
    p_extract.add_argument("-o", "--output", required=True, help="Output directory")
    p_extract.add_argument("--strategy", default="auto",
                           choices=["auto", "grid", "contours", "components"])
    p_extract.add_argument("--cell-size", default=None,
                           help="Grid cell size (e.g. '32' or '32x48')")
    p_extract.add_argument("--min-size", type=int, default=8,
                           help="Min sprite dimension (px)")
    p_extract.add_argument("--deduplicate", action="store_true", default=True,
                           help="Remove duplicate sprites")
    p_extract.add_argument("--no-deduplicate", action="store_false", dest="deduplicate")
    p_extract.add_argument("--recursive", "-r", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    # -- clean --
    p_clean = sub.add_parser("clean", help="Clean sprites to pixel-perfect quality")
    p_clean.add_argument("inputs", nargs="+", help="Directory of sprite images")
    p_clean.add_argument("-o", "--output", required=True, help="Output directory")
    p_clean.add_argument("--max-core-diff", type=float, default=8.0,
                         help="Max acceptable core diff %% (default: 8)")
    p_clean.add_argument("--upscale-target", type=int, default=256,
                         help="Target size for upscaled training version")
    p_clean.add_argument("--no-hough", action="store_true",
                         help="Disable Hough grid detection")
    p_clean.set_defaults(func=cmd_clean)

    # -- tag --
    p_tag = sub.add_parser("tag", help="Auto-tag cleaned sprites")
    p_tag.add_argument("inputs", nargs="+", help="Directory of cleaned sprite images")
    p_tag.add_argument("-o", "--output", default=None,
                       help="Output JSONL path (default: <input>/tags.jsonl)")
    p_tag.add_argument("--grid-size", type=int, default=32,
                       help="Grid size for tagging (default: 32)")
    p_tag.add_argument("--source-set", default=None,
                       help="Source tileset name (e.g. 'dcss', 'lpc')")
    p_tag.set_defaults(func=cmd_tag)

    # -- prompts --
    p_prompts = sub.add_parser("prompts", help="Generate unpixelization prompts")
    p_prompts.add_argument("inputs", nargs="+", help="Manifest JSONL path")
    p_prompts.add_argument("-o", "--output", required=True, help="Output directory")
    p_prompts.add_argument("--styles", default=None,
                           help="Comma-separated styles (photorealistic,digital_painting,...)")
    p_prompts.set_defaults(func=cmd_prompts)

    # -- caption --
    p_caption = sub.add_parser("caption", help="Describe sprites using a VLM")
    p_caption.add_argument("inputs", nargs="+", help="Directory of sprite images")
    p_caption.add_argument("--backend", default="claude",
                           choices=["claude", "openai", "local", "manual"],
                           help="VLM backend (default: claude)")
    p_caption.add_argument("--model", default=None,
                           help="Override model name for the backend")
    p_caption.add_argument("--cache-dir", default=None,
                           help="Cache directory for captions")
    p_caption.add_argument("--rate-limit", type=float, default=0.5,
                           help="Seconds between API calls (default: 0.5)")
    p_caption.set_defaults(func=cmd_caption)

    # -- generate --
    p_gen = sub.add_parser("generate", help="Generate high-res counterpart images")
    p_gen.add_argument("inputs", nargs="+", help="Manifest JSONL path")
    p_gen.add_argument("-o", "--output", required=True, help="Output directory")
    p_gen.add_argument("--backend", default="comfyui",
                       choices=["comfyui", "a1111", "openai", "replicate", "manual"],
                       help="Generation backend (default: comfyui)")
    p_gen.add_argument("--model", default=None,
                       help="Override model name/checkpoint")
    p_gen.add_argument("--width", type=int, default=1024, help="Image width")
    p_gen.add_argument("--height", type=int, default=1024, help="Image height")
    p_gen.add_argument("--num-images", type=int, default=1,
                       help="Images per prompt")
    p_gen.add_argument("--guidance-scale", type=float, default=7.5)
    p_gen.add_argument("--steps", type=int, default=30,
                       help="Inference steps")
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--styles", default=None,
                       help="Comma-separated styles to generate")
    p_gen.add_argument("--rate-limit", type=float, default=1.0,
                       help="Seconds between API calls")
    p_gen.add_argument("--regenerate", action="store_true",
                       help="Regenerate existing images")
    p_gen.set_defaults(func=cmd_generate)

    # -- package --
    p_package = sub.add_parser("package", help="Package pairs into training dataset")
    p_package.add_argument("inputs", nargs="+", help="Manifest JSONL path")
    p_package.add_argument("-o", "--output", required=True, help="Output directory")
    p_package.add_argument("--val-ratio", type=float, default=0.1)
    p_package.add_argument("--test-ratio", type=float, default=0.1)
    p_package.set_defaults(func=cmd_package)

    # -- train --
    p_train = sub.add_parser("train", help="Train a pixelization model (LoRA)")
    p_train.add_argument("--config", default=None,
                         help="Training config JSON file")
    p_train.add_argument("--manifest", default=None,
                         help="Dataset manifest.jsonl path")
    p_train.add_argument("--generated-dir", default=None,
                         help="Directory of generated high-res images")
    p_train.add_argument("--train-output", default="training_output",
                         help="Output directory for checkpoints")
    p_train.add_argument("--base-model", default=None,
                         help="HuggingFace base model ID")
    p_train.add_argument("--learning-rate", type=float, default=None)
    p_train.add_argument("--lora-rank", type=int, default=None)
    p_train.add_argument("--max-train-steps", type=int, default=None)
    p_train.add_argument("--dry-run", action="store_true",
                         help="Print config without training")
    p_train.set_defaults(func=cmd_train)

    # -- crawl-descriptions --
    p_crawl = sub.add_parser("crawl-descriptions",
                              help="Datamine DCSS tile-description pairs for curation")
    p_crawl.add_argument("-o", "--output", required=True,
                         help="Output directory")
    p_crawl.add_argument("--tiles-dir", default=None,
                         help="Path to DCSS tiles release dir")
    p_crawl.add_argument("--crawl-dir", default=None,
                         help="Path to crawl source (for description files)")
    p_crawl.add_argument("--fetch", action="store_true",
                         help="Auto-clone tiles + crawl repos")
    p_crawl.add_argument("--categories", nargs="+", default=None,
                         help="Only these tile categories (e.g. item mon)")
    p_crawl.add_argument("--no-embed", action="store_true",
                         help="Don't embed images in HTML gallery")
    p_crawl.add_argument("--selections", default=None,
                         help="Path to curation selections JSON to apply")
    p_crawl.set_defaults(func=cmd_crawl_descriptions)

    # -- run (full pipeline) --
    p_run = sub.add_parser("run",
                           help="Run full pipeline: extract → clean → tag → caption → prompts")
    p_run.add_argument("inputs", nargs="+", help="Image files or directories")
    p_run.add_argument("-o", "--output", default=None,
                       help="Base output directory (default: data/)")
    p_run.add_argument("--strategy", default="auto",
                       choices=["auto", "grid", "contours", "components"])
    p_run.add_argument("--source-set", default=None,
                       help="Source tileset name")
    p_run.add_argument("--styles", default=None,
                       help="Comma-separated generation styles")
    p_run.add_argument("--captioner-backend", default=None,
                       choices=["claude", "openai", "local", "manual", "none"],
                       help="VLM captioning backend (default: none/placeholder)")
    p_run.add_argument("--no-hough", action="store_true")
    p_run.add_argument("--recursive", "-r", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.debug)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
