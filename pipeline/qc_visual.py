"""Visual QC output for pixel art pipeline.

Generates inspection-friendly images:
  - Blown-up 1x pixel art with red grid overlay
  - Side-by-side comparison: original | grid overlay | reconstruction
  - Per-image metrics overlay
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Grid line colour (red) and width
GRID_COLOR = (255, 0, 0)
GRID_ALPHA = 180  # 0-255


def render_pixel_grid(
    image_1x: np.ndarray,
    scale: int = 16,
    grid_color: Tuple[int, int, int] = GRID_COLOR,
    grid_width: int = 1,
) -> np.ndarray:
    """Blow up a 1x pixel art image and overlay a red pixel grid.

    Args:
        image_1x: The native-resolution pixel art (H x W x 3 or 4).
        scale: How much to magnify each pixel.
        grid_color: RGB colour for the grid lines.
        grid_width: Width of grid lines in output pixels.

    Returns:
        RGB numpy array of the blown-up image with grid overlay.
    """
    h, w = image_1x.shape[:2]
    channels = image_1x.shape[2] if len(image_1x.shape) == 3 else 1

    # Nearest-neighbor upscale
    big = cv2.resize(image_1x, (w * scale, h * scale),
                     interpolation=cv2.INTER_NEAREST)

    # Ensure RGB (drop alpha for the grid overlay)
    if channels == 4:
        # Composite on white background for visibility
        alpha = big[:, :, 3:4].astype(np.float32) / 255.0
        rgb = big[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 220.0)  # light grey background
        big_rgb = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    else:
        big_rgb = big.copy()

    # Draw grid lines
    out_h, out_w = big_rgb.shape[:2]
    for y in range(0, out_h + 1, scale):
        y_clamped = min(y, out_h - 1)
        big_rgb[max(0, y_clamped - grid_width + 1):y_clamped + 1, :] = grid_color
    for x in range(0, out_w + 1, scale):
        x_clamped = min(x, out_w - 1)
        big_rgb[:, max(0, x_clamped - grid_width + 1):x_clamped + 1] = grid_color

    return big_rgb


def render_source_grid_overlay(
    source_img: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int] = (0, 0),
    grid_color: Tuple[int, int, int] = GRID_COLOR,
) -> np.ndarray:
    """Overlay the detected grid on the original source image.

    This shows if the grid detection aligned correctly with the actual
    pixel boundaries.
    """
    overlay = source_img[:, :, :3].copy()
    h, w = overlay.shape[:2]
    ox, oy = offset

    # Vertical lines
    for x in range(ox, w, cell_size):
        cv2.line(overlay, (x, 0), (x, h - 1), grid_color, 1)
    # Horizontal lines
    for y in range(oy, h, cell_size):
        cv2.line(overlay, (0, y), (w - 1, y), grid_color, 1)

    return overlay


def render_qc_panel(
    source_img: np.ndarray,
    image_1x: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    metrics: dict,
    source_name: str = "",
    max_panel_height: int = 512,
) -> np.ndarray:
    """Create a full QC panel: source with grid | blown-up 1x with grid | metrics.

    Args:
        source_img: Original source image (RGB).
        image_1x: Cleaned 1x pixel art (RGB or RGBA).
        cell_size: Detected cell size.
        offset: Grid offset.
        metrics: Quality metrics dict.
        source_name: Filename for labeling.
        max_panel_height: Maximum height for each panel section.

    Returns:
        RGB numpy array of the composite QC image.
    """
    h_src, w_src = source_img.shape[:2]
    h_1x, w_1x = image_1x.shape[:2]

    # Scale factor for the blown-up view
    pixel_scale = max(4, min(32, max_panel_height // max(h_1x, 1)))

    # Panel 1: source with grid overlay
    src_overlay = render_source_grid_overlay(source_img, cell_size, offset)
    # Resize source panel to fit max_panel_height
    src_scale = max_panel_height / max(h_src, 1)
    if src_scale < 1.0:
        new_w = int(w_src * src_scale)
        new_h = int(h_src * src_scale)
        src_panel = cv2.resize(src_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        src_panel = src_overlay

    # Panel 2: blown-up 1x with pixel grid
    pixel_panel = render_pixel_grid(image_1x, scale=pixel_scale)
    # Resize if too tall
    pp_h, pp_w = pixel_panel.shape[:2]
    if pp_h > max_panel_height:
        pp_scale = max_panel_height / pp_h
        pixel_panel = cv2.resize(
            pixel_panel,
            (int(pp_w * pp_scale), int(pp_h * pp_scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    # Panel 3: Reconstruction upscaled to match source (no grid)
    rec_up = cv2.resize(image_1x[:, :, :3], (w_src, h_src),
                        interpolation=cv2.INTER_NEAREST)
    if src_scale < 1.0:
        rec_panel = cv2.resize(rec_up, (int(w_src * src_scale), int(h_src * src_scale)),
                               interpolation=cv2.INTER_AREA)
    else:
        rec_panel = rec_up

    # Normalize all panels to same height
    target_h = max(src_panel.shape[0], pixel_panel.shape[0], rec_panel.shape[0])

    def pad_to_height(img, target):
        h, w = img.shape[:2]
        if h >= target:
            return img[:target]
        pad = np.full((target - h, w, 3), 40, dtype=np.uint8)
        return np.vstack([img, pad])

    src_panel = pad_to_height(src_panel, target_h)
    pixel_panel = pad_to_height(pixel_panel, target_h)
    rec_panel = pad_to_height(rec_panel, target_h)

    # Add thin separator columns
    sep = np.full((target_h, 3, 3), 128, dtype=np.uint8)

    composite = np.hstack([src_panel, sep, pixel_panel, sep, rec_panel])

    # Add metrics text bar at bottom
    bar_h = 60
    bar = np.full((bar_h, composite.shape[1], 3), 30, dtype=np.uint8)

    # Render text with OpenCV
    variance = metrics.get("intra_cell_variance", -1)
    core_diff = metrics.get("percent_diff_core", -1)
    tol_diff = metrics.get("percent_diff_core_tolerant", core_diff)
    method = metrics.get("method", "reconstructor")

    lines = [
        f"{source_name}  |  cell={cell_size}px  offset=({offset[0]},{offset[1]})  "
        f"|  1x: {h_1x}x{w_1x}  |  method: {method}",
        f"intra-cell var: {variance:.1f}  |  core_diff: {core_diff:.1f}%  "
        f"|  tolerant_diff: {tol_diff:.1f}%",
    ]

    font_scale = 0.45
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (10, 18 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([composite, bar])


def save_qc_image(
    source_img: np.ndarray,
    image_1x: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    metrics: dict,
    output_path: Path,
    source_name: str = "",
) -> Path:
    """Generate and save a QC panel image.

    Returns the output path.
    """
    panel = render_qc_panel(
        source_img, image_1x, cell_size, offset, metrics,
        source_name=source_name,
    )
    Image.fromarray(panel).save(output_path)
    logger.info("QC image saved: %s", output_path)
    return output_path


def save_pixel_grid(
    image_1x: np.ndarray,
    output_path: Path,
    scale: int = 16,
) -> Path:
    """Save just the blown-up pixel art with grid overlay.

    Returns the output path.
    """
    grid_img = render_pixel_grid(image_1x, scale=scale)
    Image.fromarray(grid_img).save(output_path)
    return output_path
