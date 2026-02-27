"""End-to-end tests for the pixel art pipeline.

Tests the full clean → tag → QC flow on synthetic sprites and verifies
that the pipeline produces correct results.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.cleaner import (
    CleanedAsset,
    _detect_nearest_neighbor_scale,
    _grid_quality_score,
    clean_sprite,
)
from pipeline.qc_visual import render_pixel_grid, save_qc_image
from pipeline.tagger import (
    auto_tag,
    detect_facing,
    detect_outline,
    detect_shading,
    detect_subject,
    extract_palette,
)
from pixel_perfecter.reconstructor import (
    PixelArtReconstructor,
    _compute_intra_cell_variance,
)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def _make_nn_upscaled_sprite(cell_size: int = 8, grid_w: int = 16, grid_h: int = 16) -> np.ndarray:
    """Create a clean NN-upscaled pixel art sprite (perfect grid).

    Returns an RGB image of size (grid_h * cell_size, grid_w * cell_size, 3).
    """
    rng = np.random.RandomState(42)
    art_1x = rng.randint(0, 256, (grid_h, grid_w, 3), dtype=np.uint8)
    # NN upscale
    big = np.repeat(np.repeat(art_1x, cell_size, axis=0), cell_size, axis=1)
    return big


def _make_checkerboard(size: int = 32, block: int = 4) -> np.ndarray:
    """Create an RGBA checkerboard."""
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    colors = [
        (32, 48, 112, 255),
        (240, 200, 96, 255),
        (20, 20, 24, 255),
        (220, 80, 92, 255),
    ]
    for y in range(size):
        for x in range(size):
            idx = ((x // block) + (y // block)) % len(colors)
            pixels[y, x] = colors[idx]
    return pixels


def _make_simple_pixel_art(cell_size: int = 32) -> np.ndarray:
    """Create a simple 16x16 pixel art sprite NN-upscaled to 1024x1024-ish.

    A centred circle shape on a solid background.
    """
    grid_size = 1024 // cell_size
    art = np.full((grid_size, grid_size, 3), (200, 180, 140), dtype=np.uint8)  # bg
    cy, cx = grid_size // 2, grid_size // 2
    radius = grid_size // 3
    for y in range(grid_size):
        for x in range(grid_size):
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                art[y, x] = (60, 120, 200)

    # NN upscale to full size
    big = np.repeat(np.repeat(art, cell_size, axis=0), cell_size, axis=1)
    return big


def _make_soft_pixel_art(cell_size: int = 32) -> np.ndarray:
    """Create AI-style pixel art: NN-upscaled then softened with blur.

    Simulates the anti-aliased look of AI-generated pixel art.
    """
    clean = _make_simple_pixel_art(cell_size)
    # Apply slight blur to simulate AI rendering
    soft = cv2.GaussianBlur(clean, (3, 3), 1.0)
    return soft


# ---------------------------------------------------------------------------
# Tests: NN detection
# ---------------------------------------------------------------------------


class TestNNDetection:
    def test_detects_clean_nn_upscale(self):
        img = _make_nn_upscaled_sprite(cell_size=8, grid_w=16, grid_h=16)
        scale = _detect_nearest_neighbor_scale(img)
        assert scale == 8, f"Expected NN scale 8, got {scale}"

    def test_no_nn_on_real_gradients(self):
        """Image with genuine color gradients should NOT be detected as NN."""
        # Create a gradient image - not pixel art at all
        rng = np.random.RandomState(42)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                img[y, x] = (x, y, (x + y) // 2)
        # Add some noise to break uniformity
        noise = rng.randint(-5, 6, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        scale = _detect_nearest_neighbor_scale(img)
        assert scale == 1, f"Gradient image should not be NN, got {scale}"

    def test_no_nn_on_natural_image(self):
        """A random noise image should not be detected as NN."""
        rng = np.random.RandomState(99)
        img = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        scale = _detect_nearest_neighbor_scale(img)
        assert scale == 1, f"Noise image should not be NN, got {scale}"


# ---------------------------------------------------------------------------
# Tests: Intra-cell variance
# ---------------------------------------------------------------------------


class TestIntraCellVariance:
    def test_perfect_grid_zero_variance(self):
        img = _make_nn_upscaled_sprite(cell_size=8)
        var = _compute_intra_cell_variance(img, 8, (0, 0))
        assert var == 0.0, f"Perfect NN grid should have 0 variance, got {var}"

    def test_wrong_size_higher_variance(self):
        img = _make_nn_upscaled_sprite(cell_size=8)
        var_correct = _compute_intra_cell_variance(img, 8, (0, 0))
        var_wrong = _compute_intra_cell_variance(img, 7, (0, 0))
        assert var_wrong > var_correct, "Wrong cell size should have higher variance"


# ---------------------------------------------------------------------------
# Tests: Grid quality score
# ---------------------------------------------------------------------------


class TestGridQualityScore:
    def test_correct_size_scores_highest(self):
        img = _make_simple_pixel_art(cell_size=32)
        scores = {}
        for cs in [8, 16, 32, 64]:
            scores[cs] = _grid_quality_score(img, cs)
        assert scores[32] >= scores[16], f"cell=32 should score >= cell=16, got {scores}"
        assert scores[32] >= scores[8], f"cell=32 should score >= cell=8, got {scores}"


# ---------------------------------------------------------------------------
# Tests: Clean sprite pipeline
# ---------------------------------------------------------------------------


class TestCleanSprite:
    def test_nn_fast_path(self):
        """Clean NN-upscaled sprite should take the fast path."""
        img = _make_nn_upscaled_sprite(cell_size=8, grid_w=16, grid_h=16)
        asset = clean_sprite(img, source_path="test_nn.png")
        assert asset is not None, "NN sprite should not be rejected"
        assert asset.image_1x.shape[:2] == (16, 16), f"Expected 16x16 1x, got {asset.image_1x.shape}"
        assert asset.metrics.get("method") == "nn_fast_path"

    def test_simple_pixel_art(self):
        """Simple synthetic pixel art should be cleaned successfully."""
        img = _make_simple_pixel_art(cell_size=32)
        asset = clean_sprite(img, source_path="test_simple.png", use_hough=True)
        assert asset is not None, "Simple pixel art should not be rejected"
        assert asset.image_1x.shape[0] > 4, "1x output should have reasonable size"
        assert asset.image_1x.shape[1] > 4

    def test_soft_pixel_art(self):
        """Blurred (AI-style) pixel art should still be cleaned."""
        img = _make_soft_pixel_art(cell_size=32)
        asset = clean_sprite(img, source_path="test_soft.png", use_hough=True)
        assert asset is not None, "Soft pixel art should not be rejected"

    def test_checkerboard_rgba(self):
        """RGBA checkerboard should work."""
        img = _make_checkerboard(size=32, block=4)
        asset = clean_sprite(img, source_path="test_checker.png")
        assert asset is not None, "Checkerboard should not be rejected"

    def test_tiny_image_rejected(self):
        """Very small images should be rejected."""
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        asset = clean_sprite(img, source_path="tiny.png")
        assert asset is None, "2x2 image should be rejected"


# ---------------------------------------------------------------------------
# Tests: Tagger
# ---------------------------------------------------------------------------


class TestTagger:
    def _make_simple_tagged_sprite(self):
        """A simple sprite for tagging tests."""
        # 16x16 pixel art with clear outline
        art = np.full((16, 16, 4), (180, 180, 180, 255), dtype=np.uint8)
        # Black outline
        art[0, :] = art[-1, :] = art[:, 0] = art[:, -1] = (0, 0, 0, 255)
        # Transparent corners
        art[0, 0, 3] = art[0, -1, 3] = art[-1, 0, 3] = art[-1, -1, 3] = 0
        # Interior blue
        art[3:13, 3:13] = (40, 80, 200, 255)
        return art

    def test_extract_palette(self):
        art = self._make_simple_tagged_sprite()
        palette = extract_palette(art)
        assert len(palette) > 0, "Should extract at least one colour"
        assert len(palette) <= 256, "Should not exceed max_colors"

    def test_detect_outline_black(self):
        """Sprite with a clear black outline around a transparent shape."""
        from pipeline.config import OutlineType
        # 32x32 sprite: transparent bg, black outline, colored interior
        art = np.zeros((32, 32, 4), dtype=np.uint8)  # fully transparent
        # Draw a filled circle with black outline
        for y in range(32):
            for x in range(32):
                dist = ((x - 16) ** 2 + (y - 16) ** 2) ** 0.5
                if dist < 10:
                    art[y, x] = (120, 160, 200, 255)  # interior
                elif dist < 12:
                    art[y, x] = (10, 10, 10, 255)  # black outline
        outline = detect_outline(art)
        assert outline in (OutlineType.BLACK, OutlineType.BLACK_2PX), \
            f"Circle with black border should be detected as black outline, got {outline}"

    def test_detect_subject_from_filename(self):
        from pipeline.config import SubjectType
        art = self._make_simple_tagged_sprite()
        assert detect_subject(art, "Pixel Art Sword.png") == SubjectType.ITEM
        assert detect_subject(art, "retro_character_walk.png") == SubjectType.CHARACTER
        assert detect_subject(art, "dragon_boss.png") == SubjectType.MONSTER
        assert detect_subject(art, "smiley_icon.png") == SubjectType.ICON

    def test_detect_facing_symmetric(self):
        """Symmetric sprites should be tagged as front-facing."""
        # Create perfectly symmetric sprite
        art = np.full((16, 16, 4), (100, 100, 100, 255), dtype=np.uint8)
        art[:, :3] = (0, 0, 0, 255)
        art[:, -3:] = (0, 0, 0, 255)
        from pipeline.config import FacingDirection
        facing = detect_facing(art)
        assert facing in (FacingDirection.FRONT, FacingDirection.TOP_DOWN), \
            f"Symmetric sprite should be front/top_down, got {facing}"

    def test_auto_tag_full(self):
        art = self._make_simple_tagged_sprite()
        tags = auto_tag(art, grid_size=32, source_name="pixel_sword.png")
        assert tags.grid_size == 32
        assert tags.palette_count > 0
        tag_str = tags.to_tag_string()
        assert "grid:32px" in tag_str
        assert "subject:item" in tag_str  # "sword" keyword


# ---------------------------------------------------------------------------
# Tests: QC visual output
# ---------------------------------------------------------------------------


class TestQCVisual:
    def test_render_pixel_grid(self):
        art = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        grid = render_pixel_grid(art, scale=16)
        assert grid.shape == (8 * 16, 8 * 16, 3), f"Expected 128x128x3, got {grid.shape}"

    def test_save_qc_image(self, tmp_path):
        art = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = tmp_path / "qc_test.png"
        save_qc_image(source, art, cell_size=8, offset=(0, 0),
                      metrics={"intra_cell_variance": 10.0}, output_path=out)
        assert out.exists()
        img = Image.open(out)
        assert img.size[0] > 0 and img.size[1] > 0
