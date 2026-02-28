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

from pixel_perfecter.reconstructor import (
    PixelArtReconstructor,
    _compute_intra_cell_variance,
)

logger = logging.getLogger(__name__)

# Maximum acceptable intra-cell variance.  Lower means cells are more
# uniform which indicates correct grid detection.  AI-generated pixel art
# typically has variance 20-300; detailed scenes can go to 500.
# Non-pixel-art images are typically 700+.
MAX_CELL_VARIANCE = 600.0

# Minimum acceptable cell_size for the pipeline
MIN_CELL_SIZE = 4


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

    Uses a two-tier check to avoid false positives on AI-generated pixel
    art where smooth gradients can produce low ptp at small block sizes:
      - Blocks must be strictly identical (ptp=0) at >= 60% rate, AND
      - Blocks must be near-identical (ptp<=2) at >= 90% rate.
    True NN upscaling produces ~100% ptp=0 blocks; AI gradients produce
    <1% ptp=0 blocks even when ptp<=2 is high.
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

        strict_blocks = 0  # ptp = 0 (perfectly uniform)
        loose_blocks = 0   # ptp <= 2 (near-uniform)
        total_blocks = 0

        for by in range(0, sample_h - scale + 1, scale):
            for bx in range(0, sample_w - scale + 1, scale):
                block = sample[by:by + scale, bx:bx + scale]
                total_blocks += 1
                ptp = np.ptp(block)
                if ptp == 0:
                    strict_blocks += 1
                    loose_blocks += 1
                elif ptp <= 2:
                    loose_blocks += 1

        if total_blocks == 0:
            continue

        strict_ratio = strict_blocks / total_blocks
        loose_ratio = loose_blocks / total_blocks

        # True NN upscaling: most blocks are exactly identical
        # AI-generated: smooth gradients make blocks look near-uniform but
        # NOT exactly identical.
        if strict_ratio >= 0.60 and loose_ratio >= 0.90:
            best_scale = scale
            break  # largest matching scale is the true upscale factor

    return best_scale


def _downscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    """Downscale by sampling every `scale`-th pixel."""
    return img[::scale, ::scale].copy()


# Standard pixel art cell sizes to try as fallbacks.
# For a 1024x1024 image these give: 256, 128, 64, 32, 16 cells per dim.
STANDARD_CELL_SIZES = [4, 8, 16, 32, 64]


def _grid_quality_score(img: np.ndarray, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> float:
    """Score grid quality: inter-cell contrast / sqrt(intra-cell variance).

    Peaks at the correct cell size because:
    - At correct size: cells are internally uniform (low intra) but
      differ from neighbors (high inter)
    - At too-small sizes: both inter and intra are low (subdividing one pixel)
    - At too-large sizes: intra is high (spanning multiple pixels)
    """
    h, w = img.shape[:2]
    ox, oy = offset

    grid_h = (h - oy) // cell_size
    grid_w = (w - ox) // cell_size
    if grid_h < 2 or grid_w < 2:
        return 0.0

    # Compute mean color and variance per cell using vectorized ops
    means = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    total_intra_var = 0.0

    for gy in range(grid_h):
        for gx in range(grid_w):
            y0 = oy + gy * cell_size
            x0 = ox + gx * cell_size
            cell = img[y0:y0 + cell_size, x0:x0 + cell_size, :3].astype(np.float32)
            flat = cell.reshape(-1, 3)
            means[gy, gx] = np.mean(flat, axis=0)
            total_intra_var += np.mean(np.var(flat, axis=0))

    mean_intra_var = total_intra_var / (grid_h * grid_w)

    # Inter-cell contrast: mean L1 difference between adjacent cells
    h_diff = np.sum(np.abs(means[:, :-1] - means[:, 1:]), axis=-1)
    v_diff = np.sum(np.abs(means[:-1, :] - means[1:, :]), axis=-1)
    mean_inter_diff = np.mean(np.concatenate([h_diff.ravel(), v_diff.ravel()]))

    if mean_intra_var < 0.1:
        return mean_inter_diff * 100.0  # near-perfect grid
    return float(mean_inter_diff / (mean_intra_var ** 0.5))


def _pick_best_cell_size(
    rgb: np.ndarray,
    detected_size: int,
    detected_offset: Tuple[int, int],
) -> Tuple[int, Tuple[int, int]]:
    """Refine detected cell size by snapping to a nearby standard size.

    Pixel art almost always uses standard cell sizes (8, 16, 32, 64).
    If the detected size is within 25% of a standard, snap to it and
    verify with the grid quality metric.  Uses detected size as fallback.

    Returns (cell_size, offset).
    """
    h, w = rgb.shape[:2]

    # Find the nearest valid standard size
    best_std = None
    best_dist = float("inf")
    for std in STANDARD_CELL_SIZES:
        if std < MIN_CELL_SIZE or h // std < 4 or w // std < 4:
            continue
        dist = abs(detected_size - std)
        pct = dist / max(std, 1)
        if pct <= 0.25 and dist < best_dist:
            best_dist = dist
            best_std = std

    if best_std is None or best_std == detected_size:
        return detected_size, detected_offset

    # Compare quality scores: snap only if the standard size is at least
    # as good as the detected size.
    det_score = _grid_quality_score(rgb, detected_size, detected_offset)
    std_score = _grid_quality_score(rgb, best_std, (0, 0))

    # Standard size gets a 10% bonus (clean round numbers are preferred)
    if std_score * 1.10 >= det_score:
        logger.info(
            "Snapping cell_size %d → %d (quality %.2f → %.2f)",
            detected_size, best_std, det_score, std_score,
        )
        return best_std, (0, 0)

    return detected_size, detected_offset


def _extract_alpha_1x(
    alpha_channel: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Extract 1x alpha channel by thresholding cell averages."""
    oy, ox = offset[1], offset[0]
    alpha_1x = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for gy in range(grid_h):
        for gx in range(grid_w):
            cy = oy + gy * cell_size
            cx = ox + gx * cell_size
            cell_alpha = alpha_channel[
                cy:min(cy + cell_size, alpha_channel.shape[0]),
                cx:min(cx + cell_size, alpha_channel.shape[1]),
            ]
            if cell_alpha.size > 0:
                alpha_1x[gy, gx] = 255 if np.mean(cell_alpha) > 127 else 0
    return alpha_1x


def clean_sprite(
    img: np.ndarray,
    source_path: str = "",
    use_hough: bool = True,
    upscale_target: int = 256,
    max_core_diff: float = 8.0,
    max_cell_variance: float = MAX_CELL_VARIANCE,
) -> Optional[CleanedAsset]:
    """Clean a single sprite image into a pixel-perfect asset.

    Args:
        img: RGB or RGBA numpy array of the sprite.
        source_path: Path to original file (for logging).
        use_hough: Use Hough-based grid detection.
        upscale_target: Target size for the upscaled training version.
        max_core_diff: Maximum acceptable core diff percentage for strict
            comparison (used for clean NN-upscaled sprites).
        max_cell_variance: Maximum acceptable intra-cell variance.  This is
            the primary quality metric for AI-generated pixel art.

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
                          "percent_diff_core": (1 - clean_ratio) * 100,
                          "intra_cell_variance": 0.0},
                source_path=source_path,
                warnings=[],
            )

        if clean_ratio > 0.80:
            # mostly clean NN — downscale and try reconstruction on smaller image
            rgb = _downscale_nn(rgb, nn_scale)
            if alpha_channel is not None:
                alpha_channel = _downscale_nn(alpha_channel, nn_scale)
            h, w = rgb.shape[:2]
        else:
            # NN detection was a false positive (AI gradients fooled it).
            # Reset nn_scale and process original image.
            logger.info(
                "NN scale %d rejected (clean_ratio=%.1f%%), processing original for %s",
                nn_scale, clean_ratio * 100, source_path,
            )
            nn_scale = 1
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

    detected_size = rec.cell_size
    detected_offset = rec.offset if rec.offset else (0, 0)

    if detected_size is None or detected_size < 2:
        detected_size = 4
        detected_offset = (0, 0)

    # Step 2b: refine cell size using grid quality scoring against standard sizes
    cell_size, offset = _pick_best_cell_size(rgb, detected_size, detected_offset)

    if cell_size < MIN_CELL_SIZE:
        logger.debug("No valid grid detected in %s (cell_size=%s)", source_path, cell_size)
        return None

    if cell_size != detected_size:
        logger.info(
            "Refined cell_size %d → %d for %s",
            detected_size, cell_size, source_path,
        )
        # Re-run reconstruction with the refined cell size
        try:
            rec.cell_size = cell_size
            rec.offset = offset
            result = rec._empirical_pixel_reconstruction()
        except Exception as e:
            logger.warning("Re-reconstruction failed for %s: %s", source_path, e)
            return None

    # Step 3: evaluate quality using metrics from the reconstructor
    metrics = rec.last_metrics or {}

    # Ensure intra-cell variance is available (it's the primary quality metric)
    cell_variance = _compute_intra_cell_variance(rgb, cell_size, offset)
    metrics["intra_cell_variance"] = cell_variance
    metrics["grid_quality_score"] = _grid_quality_score(rgb, cell_size, offset)

    # Step 3b: if variance is too high, try all standard sizes as a fallback.
    # This helps with sprite sheets and images where the auto-detector picks
    # a non-standard size that doesn't align well.
    if cell_variance > max_cell_variance:
        best_fallback = None
        for cs in STANDARD_CELL_SIZES:
            if cs < MIN_CELL_SIZE or h // cs < 4 or w // cs < 4:
                continue
            fb_var = _compute_intra_cell_variance(rgb, cs, (0, 0))
            if fb_var < cell_variance and fb_var <= max_cell_variance:
                if best_fallback is None or fb_var < best_fallback[1]:
                    best_fallback = (cs, fb_var)

        if best_fallback is not None:
            fallback_cs, fallback_var = best_fallback
            logger.info(
                "Fallback: cell_size %d (var=%.0f) → %d (var=%.0f) for %s",
                cell_size, cell_variance, fallback_cs, fallback_var, source_path,
            )
            cell_size = fallback_cs
            offset = (0, 0)
            cell_variance = fallback_var
            # Re-run reconstruction with fallback size
            try:
                rec.cell_size = cell_size
                rec.offset = offset
                result = rec._empirical_pixel_reconstruction()
            except Exception:
                pass
            metrics["intra_cell_variance"] = cell_variance
            metrics["grid_quality_score"] = _grid_quality_score(rgb, cell_size, offset)
            metrics["method"] = "standard_fallback"

    cell_variance = metrics.get("intra_cell_variance", float("inf"))
    core_diff = metrics.get("percent_diff_core", 100.0)
    tolerant_core_diff = metrics.get("percent_diff_core_tolerant", core_diff)

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

    # Quality gate: accept based on EITHER strict diff (for clean sprites)
    # OR intra-cell variance (for AI-generated content with anti-aliasing).
    strict_pass = core_diff <= max_core_diff
    variance_pass = cell_variance <= max_cell_variance
    tolerant_pass = tolerant_core_diff <= 50.0  # within tolerance after ±20 channel

    if not strict_pass and not variance_pass:
        logger.info(
            "Rejecting %s: core_diff=%.1f%%, cell_variance=%.1f (thresholds: %.1f%%, %.1f)",
            source_path, core_diff, cell_variance, max_core_diff, max_cell_variance,
        )
        return None

    if not strict_pass:
        logger.info(
            "Accepting %s via variance metric: cell_variance=%.1f, "
            "tolerant_core_diff=%.1f%%, strict_core_diff=%.1f%%",
            source_path, cell_variance, tolerant_core_diff, core_diff,
        )
        metrics["method"] = "variance_accepted"

    # Step 4: extract 1x pixel art
    # The reconstructor returns a small image (num_cells_y x num_cells_x x 3)
    # which IS the 1x pixel art.  Convert from BGR to RGB.
    if result is None:
        return None

    image_1x = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # handle alpha: sample from the center of each cell in the original
    grid_h, grid_w = image_1x.shape[:2]
    if alpha_channel is not None:
        alpha_1x = _extract_alpha_1x(alpha_channel, cell_size, offset, grid_h, grid_w)
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
        logger.info("Cleaned %s → %s (cell=%d, variance=%.1f, core_diff=%.1f%%)",
                     fpath.name, out_1x.name,
                     asset.cell_size,
                     asset.metrics.get("intra_cell_variance", -1),
                     asset.metrics.get("percent_diff_core", -1))

    return results
