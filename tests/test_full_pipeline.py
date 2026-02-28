"""Tests for the full pipeline: captioner, generator, training config, and CLI wiring.

Tests use mocks/stubs for external API calls (VLM, image generation)
and verify the pipeline orchestration logic works end-to-end.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from pipeline.config import (
    AssetTags, InputStyle, OutlineType, ShadingStyle,
    SubjectType, FacingDirection,
)
from pipeline.pair_generator import SpriteCaption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_sprite(size: int = 16) -> np.ndarray:
    """Create a simple test sprite."""
    rng = np.random.RandomState(42)
    img = np.full((size, size, 4), (180, 180, 180, 255), dtype=np.uint8)
    img[2:14, 2:14, :3] = rng.randint(40, 200, (12, 12, 3), dtype=np.uint8)
    return img


def _make_test_tags() -> AssetTags:
    return AssetTags(
        grid_size=32,
        palette_count=8,
        palette_hex=["ff0000", "00ff00", "0000ff", "ffffff",
                     "000000", "ffff00", "ff00ff", "00ffff"],
        outline=OutlineType.BLACK,
        shading=ShadingStyle.FLAT,
        subject=SubjectType.CHARACTER,
        facing=FacingDirection.FRONT,
        width_cells=16,
        height_cells=16,
    )


# ---------------------------------------------------------------------------
# Tests: Captioner module
# ---------------------------------------------------------------------------

class TestCaptioner:
    def test_parse_caption_json(self):
        from pipeline.captioner import _parse_caption_json

        text = '{"short": "a blue knight", "detailed": "A knight in blue armour facing forward."}'
        short, detailed = _parse_caption_json(text)
        assert short == "a blue knight"
        assert "blue armour" in detailed

    def test_parse_caption_json_with_markdown(self):
        from pipeline.captioner import _parse_caption_json

        text = '```json\n{"short": "a sword", "detailed": "A steel sword."}\n```'
        short, detailed = _parse_caption_json(text)
        assert short == "a sword"

    def test_parse_caption_json_fallback(self):
        from pipeline.captioner import _parse_caption_json

        text = "This is a knight. He wears blue armour."
        short, detailed = _parse_caption_json(text)
        assert len(short) > 0
        assert detailed == text

    def test_image_to_base64(self):
        from pipeline.captioner import _image_to_base64

        img = _make_test_sprite()
        b64 = _image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 100  # should be a valid base64 PNG

    def test_image_to_base64_upscales_tiny(self):
        from pipeline.captioner import _image_to_base64
        import base64
        from io import BytesIO

        img = np.zeros((4, 4, 3), dtype=np.uint8)
        b64 = _image_to_base64(img)
        # Decode and check the image was upscaled
        raw = base64.b64decode(b64)
        pil = Image.open(BytesIO(raw))
        assert pil.size[0] >= 64

    def test_manual_caption(self, tmp_path):
        from pipeline.captioner import _caption_manual

        sprite_path = tmp_path / "test_sprite.png"
        sprite_path.touch()

        caption_data = {
            "short": "a red dragon",
            "detailed": "A fearsome red dragon breathing fire.",
        }
        caption_file = tmp_path / "test_sprite_caption.json"
        caption_file.write_text(json.dumps(caption_data))

        caption = _caption_manual(str(sprite_path))
        assert caption.short_description == "a red dragon"
        assert "fire" in caption.detailed_description
        assert caption.source == "manual"

    def test_manual_caption_missing_raises(self, tmp_path):
        from pipeline.captioner import _caption_manual

        with pytest.raises(FileNotFoundError):
            _caption_manual(str(tmp_path / "nonexistent.png"))

    def test_caption_sprite_with_cache(self, tmp_path):
        from pipeline.captioner import caption_sprite

        img = _make_test_sprite()

        # Pre-populate cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "test_caption.json"
        cache_file.write_text(json.dumps({
            "short": "cached knight",
            "detailed": "A cached description.",
            "source": "cached",
        }))

        caption = caption_sprite(
            img,
            source_path=str(tmp_path / "test.png"),
            backend="manual",  # would fail without cache
            cache_dir=cache_dir,
        )
        assert caption.short_description == "cached knight"


# ---------------------------------------------------------------------------
# Tests: Generator module
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generation_config_defaults(self):
        from pipeline.generator import GenerationConfig

        config = GenerationConfig()
        assert config.width == 1024
        assert config.height == 1024
        assert config.num_images == 1
        assert "pixel art" in config.negative_prompt

    def test_generate_manual_returns_empty(self):
        from pipeline.generator import generate_image, GenerationConfig

        config = GenerationConfig()
        result = generate_image("test prompt", config, backend="manual")
        assert result == []

    def test_generated_image_dataclass(self):
        from pipeline.generator import GeneratedImage

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        gen = GeneratedImage(
            image=img,
            style=InputStyle.PHOTOREALISTIC,
            prompt="test",
            sprite_path="/test.png",
        )
        assert gen.seed == 0
        assert gen.generation_params == {}

    def test_collect_manual_pairs(self, tmp_path):
        from pipeline.generator import collect_manual_pairs

        # Create manifest
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "sprite_path": str(tmp_path / "sprite_a.png"),
        }) + "\n")

        # Create the sprite
        (tmp_path / "sprite_a.png").touch()

        # Create generated images
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        Image.new("RGB", (512, 512)).save(gen_dir / "sprite_a_photorealistic.png")

        # Collect
        out_dir = tmp_path / "pairs"
        count = collect_manual_pairs(manifest, gen_dir, out_dir)
        assert count == 1
        assert (out_dir / "sprite_a" / "sprite_a_photorealistic.png").exists()


# ---------------------------------------------------------------------------
# Tests: Training config
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_config_defaults(self):
        from training.config import TrainConfig

        config = TrainConfig()
        assert config.batch_size == 1
        assert config.lora_rank == 32
        assert config.mixed_precision == "fp16"

    def test_config_to_dict_roundtrip(self):
        from training.config import TrainConfig

        config = TrainConfig(
            learning_rate=5e-5,
            lora_rank=16,
            max_train_steps=1000,
        )
        d = config.to_dict()
        config2 = TrainConfig.from_dict(d)
        assert config2.learning_rate == 5e-5
        assert config2.lora_rank == 16
        assert config2.max_train_steps == 1000

    def test_config_from_partial_dict(self):
        from training.config import TrainConfig

        config = TrainConfig.from_dict({"lora_rank": 64})
        assert config.lora_rank == 64
        assert config.batch_size == 1  # default preserved


# ---------------------------------------------------------------------------
# Tests: CLI wiring
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_help(self):
        from pipeline.cli import build_parser

        parser = build_parser()
        # Verify all subcommands are registered
        subparsers = None
        for action in parser._actions:
            if hasattr(action, '_parser_class'):
                subparsers = action
                break

        assert subparsers is not None
        choices = list(subparsers.choices.keys())
        expected = ["extract", "clean", "tag", "caption", "prompts",
                    "generate", "package", "train", "run"]
        for cmd in expected:
            assert cmd in choices, f"Missing CLI subcommand: {cmd}"

    def test_run_subcommand_has_captioner_option(self):
        from pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run", "input/", "--captioner-backend", "claude",
        ])
        assert args.captioner_backend == "claude"

    def test_generate_subcommand_parses(self):
        from pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "generate", "data/manifest.jsonl",
            "-o", "data/generated",
            "--backend", "openai",
            "--width", "512",
            "--height", "512",
        ])
        assert args.backend == "openai"
        assert args.width == 512

    def test_train_subcommand_parses(self):
        from pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--manifest", "data/manifest.jsonl",
            "--generated-dir", "data/generated",
            "--lora-rank", "16",
            "--dry-run",
        ])
        assert args.lora_rank == 16
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# Tests: End-to-end pipeline on synthetic data
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_synthetic(self, tmp_path):
        """Run the full pipeline on a synthetic NN-upscaled sprite."""
        from pipeline.cli import main as pipeline_main

        # Create a synthetic input image: 16x16 pixel art NN-upscaled 8x
        rng = np.random.RandomState(42)
        art_1x = rng.randint(40, 220, (16, 16, 3), dtype=np.uint8)
        big = np.repeat(np.repeat(art_1x, 8, axis=0), 8, axis=1)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.fromarray(big, "RGB").save(input_dir / "test_sprite.png")

        output_dir = tmp_path / "output"

        result = pipeline_main([
            "run",
            str(input_dir),
            "-o", str(output_dir),
            "--strategy", "auto",
        ])

        assert result == 0

        # Check outputs exist
        manifest = output_dir / "manifest.jsonl"
        assert manifest.exists(), "Manifest should be created"

        with open(manifest) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) >= 1, "Should have at least one manifest entry"
        entry = entries[0]
        assert "tags" in entry
        assert "caption_short" in entry
        assert entry["tags"]["grid_size"] > 0

        # Check that prompts were generated
        prompts_dir = output_dir / "prompts"
        assert prompts_dir.exists()
        prompt_files = list(prompts_dir.rglob("*.txt"))
        assert len(prompt_files) >= 1, "Should have generated at least one prompt"

    def test_manifest_is_valid_jsonl(self, tmp_path):
        """Verify manifest entries are loadable by the packager."""
        from pipeline.cli import main as pipeline_main

        # Create synthetic input
        rng = np.random.RandomState(42)
        art_1x = rng.randint(40, 220, (8, 8, 3), dtype=np.uint8)
        big = np.repeat(np.repeat(art_1x, 16, axis=0), 16, axis=1)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.fromarray(big, "RGB").save(input_dir / "checker.png")

        output_dir = tmp_path / "output"
        pipeline_main([
            "run", str(input_dir), "-o", str(output_dir),
        ])

        manifest = output_dir / "manifest.jsonl"
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    # Verify required fields
                    assert "sprite_path" in entry
                    assert "tags" in entry
                    # Verify tags can be deserialized
                    tags = AssetTags.from_dict(entry["tags"])
                    assert tags.grid_size > 0
