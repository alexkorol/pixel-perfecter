"""Auto-tag pixel art sprites with metadata for model training.

Detects: grid size, palette, outline type, shading style, dimensions.
Subject type and facing direction require VLM assistance (stubbed).
"""

import logging
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import (
    AssetTags,
    NAMED_PALETTES,
    OutlineType,
    ShadingStyle,
    SubjectType,
    FacingDirection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette extraction
# ---------------------------------------------------------------------------

def extract_palette(img: np.ndarray, max_colors: int = 256) -> List[str]:
    """Extract unique colours from a pixel art image (1x resolution).

    Args:
        img: RGB or RGBA numpy array. Should be at 1x (native) resolution.
        max_colors: Cap on palette size (sprites with more are likely not
                    clean pixel art).

    Returns:
        List of hex colour strings (no '#' prefix), sorted by frequency.
    """
    if img.shape[2] == 4:
        # only count opaque pixels
        alpha = img[:, :, 3]
        opaque = alpha > 127
        pixels = img[:, :, :3][opaque]
    else:
        pixels = img[:, :, :3].reshape(-1, 3)

    if len(pixels) == 0:
        return []

    # count unique colours
    # quantize slightly to handle any rounding (±1 from JPEG etc)
    pixel_tuples = [tuple(p) for p in pixels]
    counts = Counter(pixel_tuples)

    if len(counts) > max_colors:
        # probably not clean pixel art at 1x, or has gradients
        logger.debug("Palette has %d colours (> %d), may need cleaning",
                      len(counts), max_colors)

    # sort by frequency (most common first)
    sorted_colors = counts.most_common()
    hex_list = [f"{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in sorted_colors]
    return hex_list


def match_named_palette(palette: List[str], threshold: float = 0.7
                        ) -> Optional[str]:
    """Find the closest named palette to the given colour set.

    Args:
        palette: List of hex colour strings.
        threshold: Minimum fraction of sprite colours that must match
                   the named palette.

    Returns:
        Name of the best matching palette, or None.
    """
    if not palette:
        return None

    sprite_set = set(palette)
    best_name = None
    best_score = 0.0

    for name, named_colors in NAMED_PALETTES.items():
        named_set = set(named_colors)
        # how many of the sprite's colours appear in this palette?
        # allow ±2 per channel for slight quantization differences
        matched = 0
        for sc in sprite_set:
            sr, sg, sb = int(sc[:2], 16), int(sc[2:4], 16), int(sc[4:6], 16)
            for nc in named_set:
                nr, ng, nb = int(nc[:2], 16), int(nc[2:4], 16), int(nc[4:6], 16)
                if abs(sr - nr) <= 4 and abs(sg - ng) <= 4 and abs(sb - nb) <= 4:
                    matched += 1
                    break

        score = matched / len(sprite_set) if sprite_set else 0
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name
    return None


# ---------------------------------------------------------------------------
# Outline detection
# ---------------------------------------------------------------------------

def detect_outline(img: np.ndarray) -> OutlineType:
    """Detect the outline style of a pixel art sprite.

    Operates on 1x resolution images. Examines border pixels of
    non-transparent regions.

    Returns:
        OutlineType enum value.
    """
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return OutlineType.NONE

    has_alpha = img.shape[2] == 4
    rgb = img[:, :, :3]

    if has_alpha:
        opaque = img[:, :, 3] > 127
    else:
        # detect background as the most common edge colour
        border_pixels = np.concatenate([
            rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]
        ])
        bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        diff = np.sum(np.abs(rgb.astype(np.int16) - bg_color.astype(np.int16)), axis=2)
        opaque = diff > 30

    if not np.any(opaque):
        return OutlineType.NONE

    # find edge pixels: opaque pixels adjacent to transparent/background
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(opaque.astype(np.uint8), kernel, iterations=1)
    edge_mask = dilated.astype(bool) & ~opaque

    # get the colours of opaque pixels that are on the boundary
    # (opaque pixels adjacent to non-opaque)
    eroded = cv2.erode(opaque.astype(np.uint8), kernel, iterations=1)
    boundary_mask = opaque & ~eroded.astype(bool)

    boundary_pixels = rgb[boundary_mask]
    if len(boundary_pixels) == 0:
        return OutlineType.NONE

    # analyze boundary pixel colours
    mean_color = np.mean(boundary_pixels.astype(np.float32), axis=0)
    brightness = np.mean(mean_color)

    # check if boundary pixels are very dark (black outline)
    dark_count = np.sum(np.all(boundary_pixels < 40, axis=1))
    dark_ratio = dark_count / len(boundary_pixels)

    # check if boundary pixels are very bright (glowing outline)
    bright_count = np.sum(np.all(boundary_pixels > 200, axis=1))
    bright_ratio = bright_count / len(boundary_pixels)

    # check colour variance of boundary pixels
    color_std = np.std(boundary_pixels.astype(np.float32), axis=0)
    color_variance = float(np.mean(color_std))

    # how many boundary pixels are there relative to total opaque pixels?
    total_opaque = np.sum(opaque)
    boundary_count = np.sum(boundary_mask)
    boundary_ratio = boundary_count / total_opaque if total_opaque > 0 else 0

    if boundary_ratio < 0.05:
        return OutlineType.NONE

    if dark_ratio > 0.7:
        # check if it's 2px wide by examining 2-eroded boundary
        eroded2 = cv2.erode(opaque.astype(np.uint8), kernel, iterations=2)
        boundary2 = opaque & ~eroded2.astype(bool)
        boundary2_pixels = rgb[boundary2]
        dark2 = np.sum(np.all(boundary2_pixels < 40, axis=1))
        dark2_ratio = dark2 / len(boundary2_pixels) if len(boundary2_pixels) > 0 else 0
        if dark2_ratio > 0.6:
            return OutlineType.BLACK_2PX
        return OutlineType.BLACK

    if bright_ratio > 0.5 and brightness > 180:
        return OutlineType.GLOWING

    if color_variance > 40:
        # boundary pixels have varied colours — could be selective or coloured
        # check if most boundary colours match interior colours
        interior_mask = eroded.astype(bool)
        interior_pixels = rgb[interior_mask]
        if len(interior_pixels) > 0:
            # check if boundary colours are systematically darker than interior
            int_brightness = np.mean(interior_pixels.astype(np.float32))
            if brightness < int_brightness * 0.6:
                return OutlineType.COLORED
        return OutlineType.SELECTIVE

    if boundary_ratio > 0.15 and dark_ratio > 0.3:
        return OutlineType.BLACK

    return OutlineType.SELECTIVE if boundary_ratio > 0.1 else OutlineType.NONE


# ---------------------------------------------------------------------------
# Shading style detection
# ---------------------------------------------------------------------------

def detect_shading(img: np.ndarray) -> ShadingStyle:
    """Detect the shading style of a pixel art sprite.

    Operates on 1x resolution images.
    """
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return ShadingStyle.FLAT

    has_alpha = img.shape[2] == 4
    rgb = img[:, :, :3].astype(np.float32)

    if has_alpha:
        opaque = img[:, :, 3] > 127
    else:
        opaque = np.ones((h, w), dtype=bool)

    opaque_pixels = rgb[opaque]
    if len(opaque_pixels) < 4:
        return ShadingStyle.FLAT

    # count unique colours
    unique_colors = len(set(tuple(p.astype(int)) for p in opaque_pixels))

    total_opaque = len(opaque_pixels)
    colors_per_pixel = unique_colors / total_opaque

    # very few unique colours = flat shading
    if unique_colors <= 4 or colors_per_pixel < 0.01:
        return ShadingStyle.FLAT

    # check for dithering: alternating pixel pattern
    # dithering creates a checkerboard-like pattern of different colours
    dither_score = 0
    sample_count = 0

    for y in range(min(h - 1, 32)):
        for x in range(min(w - 1, 32)):
            if not opaque[y, x] or not opaque[y, x + 1]:
                continue
            if not opaque[y + 1, x]:
                continue
            sample_count += 1
            p00 = rgb[y, x]
            p01 = rgb[y, x + 1]
            p10 = rgb[y + 1, x]

            # checkerboard: pixel differs from both horizontal and vertical neighbor
            # but diagonal matches
            h_diff = np.sum(np.abs(p00 - p01))
            v_diff = np.sum(np.abs(p00 - p10))
            if h_diff > 30 and v_diff > 30:
                if y + 1 < h and x + 1 < w and opaque[y + 1, x + 1]:
                    p11 = rgb[y + 1, x + 1]
                    d_diff = np.sum(np.abs(p00 - p11))
                    if d_diff < 30:
                        dither_score += 1

    dither_ratio = dither_score / sample_count if sample_count > 0 else 0
    if dither_ratio > 0.15:
        return ShadingStyle.DITHERED

    # check for hue-shifting: do shadow/highlight colours shift hue?
    # group pixels by brightness and check hue variance
    from colorsys import rgb_to_hsv
    hsv_pixels = []
    for p in opaque_pixels[::max(1, len(opaque_pixels) // 200)]:
        r, g, b = p / 255.0
        hsv = rgb_to_hsv(r, g, b)
        hsv_pixels.append(hsv)

    if len(hsv_pixels) > 10:
        hsv_arr = np.array(hsv_pixels)
        # sort by value (brightness)
        sorted_idx = np.argsort(hsv_arr[:, 2])
        n = len(sorted_idx)
        dark_quarter = hsv_arr[sorted_idx[:n // 4]]
        bright_quarter = hsv_arr[sorted_idx[3 * n // 4:]]

        if len(dark_quarter) > 2 and len(bright_quarter) > 2:
            dark_hue_mean = np.mean(dark_quarter[:, 0])
            bright_hue_mean = np.mean(bright_quarter[:, 0])
            hue_shift = abs(dark_hue_mean - bright_hue_mean)
            # wrap around hue circle
            hue_shift = min(hue_shift, 1.0 - hue_shift)
            if hue_shift > 0.05:
                return ShadingStyle.HUE_SHIFTED

    # check for pillow/cel shading: distinct colour bands with hard edges
    # this is basically: few unique colours but more than flat, with smooth regions
    if 4 < unique_colors <= 24 and colors_per_pixel < 0.1:
        return ShadingStyle.PILLOW

    # if many smooth gradations, it's likely anti-aliased
    if unique_colors > 24:
        return ShadingStyle.ANTI_ALIASED

    return ShadingStyle.FLAT


# ---------------------------------------------------------------------------
# Transparency detection
# ---------------------------------------------------------------------------

def has_transparent_background(img: np.ndarray) -> bool:
    """Check if the image has a transparent background."""
    if img.shape[2] != 4:
        return False
    alpha = img[:, :, 3]
    transparent_ratio = np.mean(alpha < 30)
    return transparent_ratio > 0.1


# ---------------------------------------------------------------------------
# Subject type detection (heuristic, no VLM)
# ---------------------------------------------------------------------------

def detect_subject(img_1x: np.ndarray, source_name: str = "") -> SubjectType:
    """Detect the subject type from the 1x pixel art image.

    Uses heuristics based on:
      - Image dimensions / aspect ratio
      - Opaque region shape and fill ratio
      - Source filename hints
      - Color distribution patterns

    Args:
        img_1x: 1x resolution pixel art (RGB or RGBA).
        source_name: Original filename for keyword hints.

    Returns:
        Best-guess SubjectType.
    """
    h, w = img_1x.shape[:2]
    has_alpha = img_1x.shape[2] == 4

    # --- Filename keyword hints (highest confidence) ---
    name_lower = source_name.lower()
    keyword_map = {
        SubjectType.CHARACTER: [
            "character", "player", "hero", "npc", "person", "avatar",
            "warrior", "mage", "knight", "rogue", "wizard", "archer",
            "human", "man", "woman", "girl", "boy",
        ],
        SubjectType.MONSTER: [
            "monster", "enemy", "boss", "creature", "dragon", "slime",
            "demon", "undead", "zombie", "skeleton", "beast", "goblin",
            "orc", "troll", "wyvern", "giant",
        ],
        SubjectType.ITEM: [
            "sword", "shield", "weapon", "armor", "potion", "item",
            "boot", "boots", "helm", "helmet", "ring", "amulet",
            "staff", "wand", "bow", "axe", "hammer", "gem", "key",
            "coin", "gold", "treasure", "medal", "star",
        ],
        SubjectType.TILE: [
            "tile", "floor", "wall", "ground", "terrain", "grass",
            "stone", "brick", "wood", "water", "lava",
        ],
        SubjectType.ICON: [
            "icon", "button", "ui_", "hud", "heart", "triangle",
            "circle", "square", "smiley", "emoji",
        ],
        SubjectType.ENVIRONMENT: [
            "tree", "house", "building", "castle", "bridge",
            "mountain", "cloud", "sun", "moon", "rock",
        ],
        SubjectType.EFFECT: [
            "effect", "particle", "explosion", "fire", "spark",
            "smoke", "magic", "spell", "aura",
        ],
    }

    for subject_type, keywords in keyword_map.items():
        for kw in keywords:
            if kw in name_lower:
                return subject_type

    # --- Shape-based heuristics ---
    if has_alpha:
        opaque = img_1x[:, :, 3] > 127
    else:
        # Use border color as background
        rgb = img_1x[:, :, :3]
        border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0).astype(np.uint8)
        diff = np.sum(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)), axis=2)
        opaque = diff > 30

    total_pixels = h * w
    opaque_count = np.sum(opaque)
    fill_ratio = opaque_count / total_pixels if total_pixels > 0 else 0
    aspect_ratio = w / h if h > 0 else 1.0

    # Find bounding box of opaque region
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    if not np.any(rows) or not np.any(cols):
        return SubjectType.UNKNOWN

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    bbox_aspect = bbox_w / bbox_h if bbox_h > 0 else 1.0
    bbox_fill = opaque_count / (bbox_h * bbox_w) if bbox_h * bbox_w > 0 else 0

    # Tiles: fill most of the canvas uniformly
    if fill_ratio > 0.85 and 0.8 < aspect_ratio < 1.2:
        return SubjectType.TILE

    # Icons: small, compact, often symmetric, high fill ratio
    if max(h, w) <= 24 and bbox_fill > 0.5:
        return SubjectType.ICON

    # Characters/monsters: taller than wide, moderate fill
    if bbox_aspect < 0.8 and bbox_h > 16:
        return SubjectType.CHARACTER

    # Items: small-ish objects, varied aspect ratios
    if max(bbox_h, bbox_w) < 32 and fill_ratio < 0.5:
        return SubjectType.ITEM

    # Wide objects are often environment or UI
    if bbox_aspect > 1.5:
        return SubjectType.ENVIRONMENT

    return SubjectType.UNKNOWN


# ---------------------------------------------------------------------------
# Facing direction detection (heuristic)
# ---------------------------------------------------------------------------

def detect_facing(img_1x: np.ndarray) -> FacingDirection:
    """Detect facing direction from the 1x pixel art image.

    Uses left/right asymmetry and vertical position of detail to infer
    the facing direction.

    Args:
        img_1x: 1x resolution pixel art (RGB or RGBA).

    Returns:
        Best-guess FacingDirection.
    """
    h, w = img_1x.shape[:2]
    if h < 4 or w < 4:
        return FacingDirection.NA

    has_alpha = img_1x.shape[2] == 4
    rgb = img_1x[:, :, :3].astype(np.float32)

    if has_alpha:
        opaque = img_1x[:, :, 3] > 127
    else:
        border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border.astype(np.float32), axis=0)
        diff = np.sum(np.abs(rgb - bg), axis=2)
        opaque = diff > 30

    if not np.any(opaque):
        return FacingDirection.NA

    # Check symmetry: compare left half to flipped right half
    mid_x = w // 2
    if mid_x < 2:
        return FacingDirection.NA

    left_half = rgb[:, :mid_x]
    right_half = np.flip(rgb[:, -mid_x:], axis=1)

    left_opaque = opaque[:, :mid_x]
    right_opaque = np.flip(opaque[:, -mid_x:], axis=1)

    # Compare opaque regions
    left_mass = np.sum(left_opaque)
    right_mass = np.sum(right_opaque)
    total_mass = left_mass + right_mass

    if total_mass < 4:
        return FacingDirection.NA

    # Compute asymmetry
    mass_asymmetry = (right_mass - left_mass) / total_mass if total_mass > 0 else 0

    # Color difference between left and right (in opaque regions)
    both_opaque = left_opaque & right_opaque
    if np.sum(both_opaque) > 4:
        color_diff = np.mean(np.abs(left_half[both_opaque] - right_half[both_opaque]))
    else:
        color_diff = 0

    # High symmetry → front view
    symmetry = 1.0 - (color_diff / 255.0) if color_diff < 255 else 0
    if symmetry > 0.85 and abs(mass_asymmetry) < 0.1:
        return FacingDirection.FRONT

    # Check for top-down indicators: center of mass in the middle vertically
    rows_with_content = np.where(np.any(opaque, axis=1))[0]
    if len(rows_with_content) > 0:
        vertical_center = np.mean(rows_with_content) / h
        # Top-down sprites tend to have content evenly distributed
        if abs(vertical_center - 0.5) < 0.1 and symmetry > 0.7:
            return FacingDirection.TOP_DOWN

    # Strong asymmetry → facing left or right
    if abs(mass_asymmetry) > 0.1 or color_diff > 20:
        # Determine which side has more "detail" (higher color variance)
        left_var = np.var(rgb[:, :mid_x][left_opaque]) if left_mass > 0 else 0
        right_var = np.var(rgb[:, -mid_x:][right_opaque]) if right_mass > 0 else 0

        # Characters typically face the direction with more detail/mass
        if mass_asymmetry > 0.05 or right_var > left_var * 1.2:
            return FacingDirection.RIGHT
        elif mass_asymmetry < -0.05 or left_var > right_var * 1.2:
            return FacingDirection.LEFT

    # Three-quarter view: slight asymmetry but front-ish
    if 0.6 < symmetry < 0.85:
        return FacingDirection.THREE_QUARTER

    return FacingDirection.FRONT


# ---------------------------------------------------------------------------
# Main auto-tagger
# ---------------------------------------------------------------------------

def auto_tag(
    img_1x: np.ndarray,
    grid_size: int,
    source_set: str = "",
    source_name: str = "",
) -> AssetTags:
    """Automatically tag a cleaned pixel art sprite.

    Args:
        img_1x: The 1x (native resolution) pixel art image, RGB or RGBA.
        grid_size: The detected grid cell size.
        source_set: Name of the source tileset (e.g. "dcss", "lpc").
        source_name: Original filename (used for keyword-based subject hints).

    Returns:
        AssetTags with all detectable fields populated.
    """
    h, w = img_1x.shape[:2]

    # palette
    palette = extract_palette(img_1x)
    palette_name = match_named_palette(palette)

    # outline
    outline = detect_outline(img_1x)

    # shading
    shading = detect_shading(img_1x)

    # transparency
    transparent = has_transparent_background(img_1x)

    # subject and facing (heuristic-based)
    subject = detect_subject(img_1x, source_name=source_name)
    facing = detect_facing(img_1x)

    tags = AssetTags(
        grid_size=grid_size,
        palette_count=len(palette),
        palette_hex=palette[:64],  # cap stored palette at 64 colours
        palette_name=palette_name,
        outline=outline,
        shading=shading,
        subject=subject,
        facing=facing,
        source_set=source_set,
        width_cells=w,
        height_cells=h,
        transparent_bg=transparent,
    )

    return tags


def batch_tag(
    sprite_dir: str,
    grid_size: int,
    source_set: str = "",
) -> List[Tuple[str, AssetTags]]:
    """Auto-tag all sprites in a directory.

    Returns list of (filename, AssetTags).
    """
    from pathlib import Path
    from PIL import Image

    results = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    sprite_path = Path(sprite_dir)

    for fpath in sorted(sprite_path.iterdir()):
        if fpath.suffix.lower() not in extensions:
            continue

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

        tags = auto_tag(img, grid_size=grid_size, source_set=source_set)
        results.append((fpath.name, tags))
        logger.debug("Tagged %s: %s", fpath.name, tags.to_tag_string())

    return results
