"""Clean extracted sprites into pixel-perfect assets using pixel-perfecter.

Takes raw extracted sprites (possibly with anti-aliasing, halo glow,
JPEG artifacts, scaling artifacts) and produces clean pixel-perfect
versions at 1x native resolution.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pixel_perfecter.reconstructor import PixelArtReconstructor

logger = logging.getLogger(__name__)


@dataclass
class CleanedAsset:
    """A pixel-perfect cleaned sprite."""
    image_1x: np.ndarray       # native resolution (1 pixel per grid cell)
    image_upscaled: np.ndarray  # nearest-neighbor upscaled for training
    cell_size: int             # detected grid cell size in source
    offset: Tuple[int, int]    # grid offset in source
    metrics: dict              # reconstruction quality metrics
    source_path: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _detect_nearest_neighbor_scale(img: np.ndarray) -> int:
    """Detect if an image has been nearest-neighbor upscaled.

    Returns the detected scale factor (1 if no upscaling detected).
    Checks for repeating pixel blocks — a telltale sign of NN upscale.
    Tests largest scales first so we find the true native resolution.
    """
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return 1

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

    # test LARGEST scale factors first — we want the true native resolution
    best_scale = 1
    for scale in [16, 12, 10, 8, 6, 5, 4, 3, 2]:
        if h % scale != 0 or w % scale != 0:
            continue
        # require at least 4 cells per dimension
        if h // scale < 4 or w // scale < 4:
            continue

        # sample a section and check if every scale x scale block is uniform
        sample_h = min(h, 64 * scale)
        sample_w = min(w, 64 * scale)
        sample = gray[:sample_h, :sample_w]

        uniform_blocks = 0
        total_blocks = 0

        for by in range(0, sample_h - scale + 1, scale):
            for bx in range(0, sample_w - scale + 1, scale):
                block = sample[by:by + scale, bx:bx + scale]
                total_blocks += 1
                if np.ptp(block) <= 2:  # peak-to-peak range
                    uniform_blocks += 1

        if total_blocks > 0 and uniform_blocks / total_blocks > 0.85:
            best_scale = scale
            break  # largest matching scale is the true upscale factor

    return best_scale


def _downscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    """Downscale by sampling every `scale`-th pixel."""
    return img[::scale, ::scale].copy()


def clean_sprite(
    img: np.ndarray,
    source_path: str = "",
    use_hough: bool = True,
    upscale_target: int = 256,
    max_core_diff: float = 8.0,
) -> Optional[CleanedAsset]:
    """Clean a single sprite image into a pixel-perfect asset.

    Args:
        img: RGB or RGBA numpy array of the sprite.
        source_path: Path to original file (for logging).
        use_hough: Use Hough-based grid detection.
        upscale_target: Target size for the upscaled training version.
        max_core_diff: Maximum acceptable core diff percentage.
            Sprites worse than this are rejected.

    Returns:
        CleanedAsset or None if cleaning failed/rejected.
    """
    has_alpha = img.shape[2] == 4
    alpha_channel = None

    if has_alpha:
        alpha_channel = img[:, :, 3]
        rgb = img[:, :, :3]
    else:
        rgb = img

    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        logger.debug("Sprite too small: %dx%d", w, h)
        return None

    # Step 1: detect if this is already an NN-upscaled sprite
    nn_scale = _detect_nearest_neighbor_scale(rgb)
    if nn_scale > 1:
        logger.info("Detected %dx NN upscale in %s", nn_scale, source_path)

    # FAST PATH: if NN upscale detected with high confidence, the 1x version
    # is just a direct downsample — no reconstruction needed.
    if nn_scale >= 2:
        image_1x = _downscale_nn(rgb, nn_scale)
        alpha_1x = None
        if alpha_channel is not None:
            alpha_1x_raw = _downscale_nn(alpha_channel, nn_scale)
            # threshold: >50% = opaque
            alpha_1x = np.where(alpha_1x_raw > 127, 255, 0).astype(np.uint8)

        # verify the downscale is clean (reconstruct and compare)
        upcheck = np.repeat(np.repeat(image_1x, nn_scale, axis=0),
                            nn_scale, axis=1)
        h_cmp = min(rgb.shape[0], upcheck.shape[0])
        w_cmp = min(rgb.shape[1], upcheck.shape[1])
        diff_pixels = np.any(
            rgb[:h_cmp, :w_cmp] != upcheck[:h_cmp, :w_cmp], axis=2
        )
        clean_ratio = 1.0 - (np.sum(diff_pixels) / diff_pixels.size)

        if clean_ratio > 0.92:
            # clean NN upscale — skip full reconstruction
            logger.info("Clean NN downscale (%.1f%% match), using fast path",
                        clean_ratio * 100)
            if alpha_1x is not None:
                image_1x = np.dstack([image_1x, alpha_1x])

            h1, w1 = image_1x.shape[:2]
            up_scale = max(1, upscale_target // max(h1, w1))
            image_up = cv2.resize(image_1x, (w1 * up_scale, h1 * up_scale),
                                  interpolation=cv2.INTER_NEAREST)

            return CleanedAsset(
                image_1x=image_1x,
                image_upscaled=image_up,
                cell_size=nn_scale,
                offset=(0, 0),
                metrics={"clean_ratio": clean_ratio, "method": "nn_fast_path",
                          "percent_diff_core": (1 - clean_ratio) * 100},
                source_path=source_path,
                warnings=[],
            )

        # not perfectly clean NN — downscale and try reconstruction
        rgb = _downscale_nn(rgb, nn_scale)
        if alpha_channel is not None:
            alpha_channel = _downscale_nn(alpha_channel, nn_scale)
        h, w = rgb.shape[:2]
    # end NN detection

    # Step 2: run pixel-perfecter reconstruction
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        rec = PixelArtReconstructor(image=bgr, debug=False)
        rec.use_hough = use_hough
        result = rec.run(mode="global")
    except Exception as e:
        logger.warning("Reconstruction failed for %s: %s", source_path, e)
        return None

    cell_size = rec.cell_size
    offset = rec.offset if rec.offset else (0, 0)

    if cell_size is None or cell_size < 1:
        logger.debug("No grid detected in %s", source_path)
        return None

    # Step 3: evaluate quality
    try:
        from pixel_perfecter.reconstructor import build_validation_diagnostics
        build_validation_diagnostics(rec)
        metrics = rec.last_metrics or {}
    except Exception:
        metrics = {}

    core_diff = metrics.get("percent_diff_core", 100.0)
    warnings = []
    if isinstance(metrics.get("warnings"), list):
        warnings = metrics["warnings"]
    elif isinstance(metrics.get("warnings"), str):
        warnings = [metrics["warnings"]] if metrics["warnings"] else []

    # If cell_size == 1 after NN downscale, the image is already at 1x
    if cell_size == 1 and nn_scale > 1:
        image_1x = rgb.copy()
        if alpha_channel is not None:
            alpha_1x = np.where(alpha_channel > 127, 255, 0).astype(np.uint8)
            image_1x = np.dstack([image_1x, alpha_1x])

        h1, w1 = image_1x.shape[:2]
        up_scale = max(1, upscale_target // max(h1, w1))
        image_up = cv2.resize(image_1x, (w1 * up_scale, h1 * up_scale),
                              interpolation=cv2.INTER_NEAREST)
        return CleanedAsset(
            image_1x=image_1x,
            image_upscaled=image_up,
            cell_size=nn_scale,
            offset=(0, 0),
            metrics=metrics,
            source_path=source_path,
            warnings=warnings,
        )

    if core_diff > max_core_diff:
        logger.info("Rejecting %s: core_diff=%.1f%% > %.1f%%",
                     source_path, core_diff, max_core_diff)
        return None

    # Step 4: extract 1x pixel art
    # result from rec is BGR, convert to RGB
    if result is None:
        return None
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # The reconstructor returns the image at full size with cells filled.
    # We want the 1x version (one pixel per cell).
    grid_h = h // cell_size
    grid_w = w // cell_size
    ox, oy = offset

    image_1x = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for gy in range(grid_h):
        for gx in range(grid_w):
            py = oy + gy * cell_size + cell_size // 2
            px = ox + gx * cell_size + cell_size // 2
            py = min(py, h - 1)
            px = min(px, w - 1)
            image_1x[gy, gx] = result_rgb[py, px]

    # handle alpha: sample from the center of each cell
    if alpha_channel is not None:
        alpha_1x = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for gy in range(grid_h):
            for gx in range(grid_w):
                cy = oy + gy * cell_size
                cx = ox + gx * cell_size
                cell_alpha = alpha_channel[
                    cy:min(cy + cell_size, alpha_channel.shape[0]),
                    cx:min(cx + cell_size, alpha_channel.shape[1])
                ]
                if cell_alpha.size > 0:
                    alpha_1x[gy, gx] = 255 if np.mean(cell_alpha) > 127 else 0
        image_1x = np.dstack([image_1x, alpha_1x])

    # account for NN downscale in reported cell size
    effective_cell = cell_size * nn_scale if nn_scale > 1 else cell_size

    # Step 5: create upscaled version for training
    h1, w1 = image_1x.shape[:2]
    scale = max(1, upscale_target // max(h1, w1))
    if scale < 1:
        scale = 1
    image_up = cv2.resize(image_1x, (w1 * scale, h1 * scale),
                          interpolation=cv2.INTER_NEAREST)

    return CleanedAsset(
        image_1x=image_1x,
        image_upscaled=image_up,
        cell_size=effective_cell,  # report original cell size
        offset=offset,
        metrics=metrics,
        source_path=source_path,
        warnings=warnings,
    )


def bulk_clean(
    sprite_dir: Path,
    output_dir: Path,
    use_hough: bool = True,
    upscale_target: int = 256,
    max_core_diff: float = 8.0,
) -> List[Tuple[Path, CleanedAsset]]:
    """Process all sprites in a directory.

    Returns list of (output_path, CleanedAsset) for successful cleanings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    files = sorted(
        p for p in sprite_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    for fpath in files:
        try:
            pil = Image.open(fpath)
            if pil.mode == "RGBA":
                img = np.array(pil)
            elif pil.mode == "P":
                img = np.array(pil.convert("RGBA"))
            else:
                img = np.array(pil.convert("RGB"))
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            continue

        asset = clean_sprite(
            img,
            source_path=str(fpath),
            use_hough=use_hough,
            upscale_target=upscale_target,
            max_core_diff=max_core_diff,
        )

        if asset is None:
            logger.info("Skipped %s (cleaning failed or rejected)", fpath.name)
            continue

        # save 1x and upscaled versions
        stem = fpath.stem
        out_1x = output_dir / f"{stem}_1x.png"
        out_up = output_dir / f"{stem}_up.png"

        channels = asset.image_1x.shape[2] if len(asset.image_1x.shape) == 3 else 1
        if channels == 4:
            Image.fromarray(asset.image_1x, "RGBA").save(out_1x)
            Image.fromarray(asset.image_upscaled, "RGBA").save(out_up)
        else:
            Image.fromarray(asset.image_1x, "RGB").save(out_1x)
            Image.fromarray(asset.image_upscaled, "RGB").save(out_up)

        results.append((out_1x, asset))
        logger.info("Cleaned %s → %s (cell=%d, core_diff=%.1f%%)",
                     fpath.name, out_1x.name,
                     asset.cell_size,
                     asset.metrics.get("percent_diff_core", -1))

    return results
