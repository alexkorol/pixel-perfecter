"""Extract individual sprites from sprite sheets, showcases, and social media images.

Handles three common layouts:
  1. Uniform grid sheets (RPG Maker, DCSS tilesets, etc.)
  2. Showcase/collage images with varying backgrounds
  3. Transparent-background sheets (connected component extraction)
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class ExtractedSprite:
    """A single sprite extracted from a larger image."""
    image: np.ndarray          # RGBA or RGB numpy array
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in source image
    source_path: str = ""
    index: int = 0             # position index in the sheet
    sprite_hash: str = ""      # perceptual hash for dedup


def _perceptual_hash(img: np.ndarray, hash_size: int = 16) -> str:
    """Simple average-hash for deduplication."""
    if img.shape[2] == 4:
        # composite on white for hashing
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        composited = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    else:
        composited = img
    gray = cv2.cvtColor(composited, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = (resized > mean).flatten()
    # pack into hex string
    byte_arr = np.packbits(bits)
    return hashlib.md5(byte_arr.tobytes()).hexdigest()[:16]


def _is_mostly_uniform(region: np.ndarray, threshold: float = 12.0) -> bool:
    """Check if a region is mostly a single colour (background)."""
    if region.size == 0:
        return True
    if region.shape[2] == 4:
        alpha = region[:, :, 3]
        if np.mean(alpha) < 30:
            return True  # mostly transparent
    std = np.std(region[:, :, :3].astype(np.float32), axis=(0, 1))
    return float(np.mean(std)) < threshold


def _detect_background_color(img: np.ndarray) -> np.ndarray:
    """Detect the most likely background colour from image borders."""
    h, w = img.shape[:2]
    border_size = max(2, min(h, w) // 20)
    # sample border strips
    strips = []
    strips.append(img[:border_size, :, :3].reshape(-1, 3))         # top
    strips.append(img[-border_size:, :, :3].reshape(-1, 3))        # bottom
    strips.append(img[:, :border_size, :3].reshape(-1, 3))         # left
    strips.append(img[:, -border_size:, :3].reshape(-1, 3))        # right
    border_pixels = np.concatenate(strips, axis=0)

    # find the most common colour via histogram binning
    # quantize to reduce noise
    quantized = (border_pixels // 8) * 8
    # find mode by hashing
    pixel_tuples = [tuple(p) for p in quantized]
    from collections import Counter
    counts = Counter(pixel_tuples)
    bg_q = np.array(counts.most_common(1)[0][0], dtype=np.uint8)

    # refine: average all border pixels within 30 of the quantized mode
    diffs = np.abs(border_pixels.astype(np.int16) - bg_q.astype(np.int16))
    close_mask = np.all(diffs < 30, axis=1)
    if np.any(close_mask):
        return np.mean(border_pixels[close_mask], axis=0).astype(np.uint8)
    return bg_q


# ---- Strategy 1: Uniform grid splitting ----

def _detect_grid_layout(img: np.ndarray) -> Optional[Tuple[int, int]]:
    """Attempt to detect a uniform grid in the image.

    Returns (cell_w, cell_h) or None if no clear grid found.
    Uses edge projection to find repeating dividers.
    """
    h, w = img.shape[:2]
    if h < 16 or w < 16:
        return None

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    results = []
    for axis, length in [(1, w), (0, h)]:
        projection = np.sum(edges, axis=axis).astype(np.float64)
        if projection.max() == 0:
            return None
        projection /= projection.max()

        # find strong lines (>50% of max)
        threshold = 0.4
        line_positions = np.where(projection > threshold)[0]
        if len(line_positions) < 3:
            results.append(None)
            continue

        # compute gaps between consecutive line positions
        # cluster nearby positions first
        clusters = []
        current_cluster = [line_positions[0]]
        for pos in line_positions[1:]:
            if pos - current_cluster[-1] <= 2:
                current_cluster.append(pos)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [pos]
        clusters.append(int(np.mean(current_cluster)))

        if len(clusters) < 3:
            results.append(None)
            continue

        gaps = np.diff(clusters)
        if len(gaps) == 0:
            results.append(None)
            continue

        median_gap = int(np.median(gaps))
        if median_gap < 8:
            results.append(None)
            continue

        # check consistency: >60% of gaps within ±15% of median
        tolerance = max(2, int(median_gap * 0.15))
        consistent = np.sum(np.abs(gaps - median_gap) <= tolerance)
        if consistent / len(gaps) < 0.6:
            results.append(None)
            continue

        results.append(median_gap)

    if results[0] is not None and results[1] is not None:
        return (results[0], results[1])
    # fall back to square cells if only one axis detected
    if results[0] is not None:
        return (results[0], results[0])
    if results[1] is not None:
        return (results[1], results[1])
    return None


def split_grid(img: np.ndarray, cell_w: int, cell_h: int,
               source_path: str = "") -> List[ExtractedSprite]:
    """Split an image on a known uniform grid."""
    h, w = img.shape[:2]
    sprites = []
    idx = 0

    # try to detect offset (grid lines / padding between cells)
    # scan first row for a thin divider line
    offset_x, offset_y = 0, 0
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

    # check if there are 1-2px divider lines
    for test_offset in range(min(4, cell_w)):
        col = test_offset
        if col >= w:
            break
        strip = gray[:, col]
        if np.std(strip.astype(np.float32)) < 10:
            offset_x = test_offset + 1
            break

    for test_offset in range(min(4, cell_h)):
        row = test_offset
        if row >= h:
            break
        strip = gray[row, :]
        if np.std(strip.astype(np.float32)) < 10:
            offset_y = test_offset + 1
            break

    y = offset_y
    while y + cell_h <= h:
        x = offset_x
        while x + cell_w <= w:
            cell = img[y:y + cell_h, x:x + cell_w].copy()
            if not _is_mostly_uniform(cell):
                sprite = ExtractedSprite(
                    image=cell,
                    bbox=(x, y, cell_w, cell_h),
                    source_path=source_path,
                    index=idx,
                    sprite_hash=_perceptual_hash(cell),
                )
                sprites.append(sprite)
            idx += 1
            x += cell_w
            # skip 1-2px divider lines
            while x < w and x < (x + 3):
                if x >= w:
                    break
                strip = gray[:min(y + cell_h, h), x]
                if np.std(strip.astype(np.float32)) < 10:
                    x += 1
                else:
                    break
        y += cell_h
        # skip divider rows
        while y < h and y < (y + 3):
            if y >= h:
                break
            strip = gray[y, :min(x + cell_w, w)]
            if np.std(strip.astype(np.float32)) < 10:
                y += 1
            else:
                break

    return sprites


# ---- Strategy 2: Contour-based extraction (showcases, collages) ----

def split_contours(img: np.ndarray, bg_color: Optional[np.ndarray] = None,
                   min_size: int = 8, padding: int = 2,
                   source_path: str = "") -> List[ExtractedSprite]:
    """Extract sprites by finding contours against a background colour."""
    if bg_color is None:
        bg_color = _detect_background_color(img)

    h, w = img.shape[:2]
    rgb = img[:, :, :3].astype(np.int16)
    diff = np.abs(rgb - bg_color.astype(np.int16))
    mask = np.any(diff > 25, axis=2).astype(np.uint8) * 255

    # handle alpha channel
    if img.shape[2] == 4:
        alpha_mask = (img[:, :, 3] > 30).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, alpha_mask)

    # clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sprites = []
    for idx, contour in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < min_size or ch < min_size:
            continue

        # add padding
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + cw + padding)
        y1 = min(h, y + ch + padding)

        cell = img[y0:y1, x0:x1].copy()

        # create alpha mask from contour
        if img.shape[2] != 4:
            alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            shifted_contour = contour - np.array([x0, y0])
            cv2.drawContours(alpha, [shifted_contour], -1, 255, cv2.FILLED)
            # dilate slightly to catch edges
            alpha = cv2.dilate(alpha, kernel, iterations=1)
            cell_rgba = np.dstack([cell[:, :, :3], alpha])
        else:
            cell_rgba = cell

        if _is_mostly_uniform(cell_rgba):
            continue

        sprite = ExtractedSprite(
            image=cell_rgba,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            source_path=source_path,
            index=idx,
            sprite_hash=_perceptual_hash(cell_rgba),
        )
        sprites.append(sprite)

    # sort by position: top-to-bottom, left-to-right
    sprites.sort(key=lambda s: (s.bbox[1] // 32, s.bbox[0]))
    for i, s in enumerate(sprites):
        s.index = i

    return sprites


# ---- Strategy 3: Connected component extraction (transparent bg) ----

def split_components(img: np.ndarray, min_size: int = 8,
                     padding: int = 2,
                     source_path: str = "") -> List[ExtractedSprite]:
    """Extract sprites from an image with transparent background."""
    if img.shape[2] != 4:
        # no alpha channel, fall back to contour method
        return split_contours(img, min_size=min_size, padding=padding,
                              source_path=source_path)

    h, w = img.shape[:2]
    alpha = img[:, :, 3]
    binary = (alpha > 20).astype(np.uint8)

    # close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels = cv2.connectedComponents(binary)

    sprites = []
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8)
        ys, xs = np.where(component_mask)
        if len(xs) == 0:
            continue

        x0 = max(0, xs.min() - padding)
        y0 = max(0, ys.min() - padding)
        x1 = min(w, xs.max() + 1 + padding)
        y1 = min(h, ys.max() + 1 + padding)

        cw, ch = x1 - x0, y1 - y0
        if cw < min_size or ch < min_size:
            continue

        cell = img[y0:y1, x0:x1].copy()

        # zero out pixels not in this component
        local_mask = component_mask[y0:y1, x0:x1]
        cell[:, :, 3] = cell[:, :, 3] * local_mask

        if _is_mostly_uniform(cell):
            continue

        sprite = ExtractedSprite(
            image=cell,
            bbox=(x0, y0, cw, ch),
            source_path=source_path,
            index=label_id - 1,
            sprite_hash=_perceptual_hash(cell),
        )
        sprites.append(sprite)

    sprites.sort(key=lambda s: (s.bbox[1] // 32, s.bbox[0]))
    for i, s in enumerate(sprites):
        s.index = i

    return sprites


# ---- Auto-detection and main entry point ----

def _has_alpha(img: np.ndarray) -> bool:
    if img.shape[2] != 4:
        return False
    alpha = img[:, :, 3]
    # significant transparency present
    return float(np.mean(alpha < 200)) > 0.05


def _estimate_strategy(img: np.ndarray) -> str:
    """Guess the best extraction strategy for this image."""
    if _has_alpha(img):
        # check if it's a grid with transparent bg
        grid = _detect_grid_layout(img)
        if grid is not None:
            return "grid"
        return "components"

    # try grid detection
    grid = _detect_grid_layout(img)
    if grid is not None:
        return "grid"

    return "contours"


def extract_sprites(
    image_path: str,
    strategy: str = "auto",
    cell_size: Optional[Tuple[int, int]] = None,
    min_size: int = 8,
    padding: int = 2,
) -> List[ExtractedSprite]:
    """Extract individual sprites from an image.

    Args:
        image_path: Path to the source image.
        strategy: "auto", "grid", "contours", or "components".
        cell_size: (w, h) for grid mode. Auto-detected if None.
        min_size: Minimum sprite dimension in pixels.
        padding: Pixels of padding around each extracted sprite.

    Returns:
        List of ExtractedSprite objects.
    """
    pil_img = Image.open(image_path)
    if pil_img.mode == "RGBA":
        img = np.array(pil_img)
    elif pil_img.mode == "P":
        pil_img = pil_img.convert("RGBA")
        img = np.array(pil_img)
    else:
        pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)

    source = str(image_path)

    if strategy == "auto":
        strategy = _estimate_strategy(img)

    if strategy == "grid":
        if cell_size is None:
            detected = _detect_grid_layout(img)
            if detected is None:
                # fall back to contour extraction
                return split_contours(img, min_size=min_size, padding=padding,
                                      source_path=source)
            cell_size = detected
        return split_grid(img, cell_size[0], cell_size[1], source_path=source)

    elif strategy == "components":
        return split_components(img, min_size=min_size, padding=padding,
                                source_path=source)

    else:  # contours
        return split_contours(img, min_size=min_size, padding=padding,
                              source_path=source)


def save_sprites(sprites: List[ExtractedSprite], output_dir: Path,
                 prefix: str = "sprite") -> List[Path]:
    """Save extracted sprites as individual PNG files.

    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for sprite in sprites:
        fname = f"{prefix}_{sprite.index:04d}.png"
        fpath = output_dir / fname
        if sprite.image.shape[2] == 4:
            pil = Image.fromarray(sprite.image, "RGBA")
        else:
            pil = Image.fromarray(sprite.image, "RGB")
        pil.save(fpath)
        saved.append(fpath)
    return saved


def deduplicate_sprites(sprites: List[ExtractedSprite],
                        hash_threshold: int = 0) -> List[ExtractedSprite]:
    """Remove duplicate sprites by perceptual hash.

    hash_threshold=0 means exact hash match only.
    """
    seen = set()
    unique = []
    for sprite in sprites:
        if sprite.sprite_hash not in seen:
            seen.add(sprite.sprite_hash)
            unique.append(sprite)
    return unique
