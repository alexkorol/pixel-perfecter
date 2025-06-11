import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple, List, Optional
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict

class PixelPerfecterResult:
    """Class to hold the result of the pixel perfecter algorithm"""
    def __init__(self, image: Image.Image, grid_size: Tuple[int, int], scale: int):
        self.image = image
        self.grid_size = grid_size  # (rows, cols)
        self.scale = scale

def detect_implied_pixels(
    image: Image.Image,
    debug_dir: Optional[str] = None,
    asset_idx: Optional[int] = None
) -> np.ndarray:
    """
    Detect implied pixels in the image using adaptive thresholding and contour detection.
    
    Args:
        image: PIL Image
        debug_dir: Optional directory for debug output
        asset_idx: Asset index for debug output
        
    Returns:
        Numpy array of detected contours
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Handle different image formats
    if image_np.ndim == 2:  # Grayscale
        gray = image_np
        alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
    elif image_np.shape[2] == 3:  # RGB
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
    elif image_np.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(image_np[..., :3], cv2.COLOR_RGB2GRAY)
        alpha = image_np[..., 3]
    else:
        logging.error(f"Unsupported image shape: {image_np.shape}")
        return None
    
    # Create alpha mask
    alpha_mask = alpha > 10  # Mask out mostly transparent areas
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to find edges
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply the alpha mask
    thresh = cv2.bitwise_and(thresh, thresh, mask=alpha_mask.astype(np.uint8) * 255)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_contour_area = 16  # Minimum area to consider
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
    
    logging.info(f"Detected {len(filtered_contours)} potential implied pixels")
    
    # Debug visualization
    if debug_dir:
        import os
        from PIL import Image, ImageDraw
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create debug image
        debug_img = Image.fromarray(image_np if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB))
        draw = ImageDraw.Draw(debug_img)
        
        # Draw contours
        for cnt in filtered_contours:
            for point in cnt:
                x, y = point[0]
                draw.rectangle([x-2, y-2, x+2, y+2], fill=(255, 0, 0))
        
        prefix = f"asset_{asset_idx}_" if asset_idx is not None else ""
        debug_img.save(os.path.join(debug_dir, f"{prefix}detected_contours.png"))
        
        # Save threshold image
        Image.fromarray(thresh).save(os.path.join(debug_dir, f"{prefix}threshold.png"))
    
    return filtered_contours

def extract_block_features(
    image: Image.Image,
    contours: List[np.ndarray]
) -> List[Dict]:
    """
    Extract features from detected blocks/contours.
    
    Args:
        image: PIL Image
        contours: List of contour arrays from OpenCV
        
    Returns:
        List of dictionaries containing block features
    """
    image_np = np.array(image)
    blocks = []
    
    for i, contour in enumerate(contours):
        # Create mask for this contour
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate average color within the contour
        if image_np.ndim == 3:
            # For RGB/RGBA images
            if image_np.shape[2] == 4:  # RGBA
                # Only consider pixels with alpha > 0
                mask_3d = np.stack([mask, mask, mask, mask], axis=2) > 0
                pixels = image_np[mask_3d].reshape(-1, 4)
                # Filter out transparent pixels
                pixels = pixels[pixels[:, 3] > 10]
                if len(pixels) == 0:
                    continue
                # Calculate median color (more robust than mean)
                color = tuple(np.median(pixels, axis=0).astype(np.uint8))
            else:  # RGB
                mask_3d = np.stack([mask, mask, mask], axis=2) > 0
                pixels = image_np[mask_3d].reshape(-1, 3)
                if len(pixels) == 0:
                    continue
                color = tuple(np.median(pixels, axis=0).astype(np.uint8))
        else:
            # For grayscale images
            pixels = image_np[mask > 0]
            if len(pixels) == 0:
                continue
            gray_value = int(np.median(pixels))
            color = (gray_value, gray_value, gray_value, 255)
        
        # Store block features
        blocks.append({
            'id': i,
            'centroid': (cx, cy),
            'bbox': (x, y, x+w, y+h),
            'dimensions': (w, h),
            'area': cv2.contourArea(contour),
            'color': color
        })
    
    logging.info(f"Extracted features for {len(blocks)} blocks")
    return blocks

def estimate_grid(
    blocks: List[Dict],
    debug_dir: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Estimate the grid parameters from block centroids using clustering.
    
    Args:
        blocks: List of block feature dictionaries
        debug_dir: Optional directory for debug output
        
    Returns:
        Tuple of (row_step, col_step, row_offset, col_offset)
    """
    if not blocks:
        logging.error("No blocks provided for grid estimation")
        return (8.0, 8.0, 0.0, 0.0)  # Default values
    
    # Extract centroids
    centroids = np.array([block['centroid'] for block in blocks])
    
    # Calculate x and y differences between adjacent centroids
    centroids_sorted_x = centroids[np.argsort(centroids[:, 0])]
    centroids_sorted_y = centroids[np.argsort(centroids[:, 1])]
    
    x_diffs = []
    y_diffs = []
    
    # Find differences between consecutive x coordinates
    for i in range(1, len(centroids_sorted_x)):
        diff = centroids_sorted_x[i, 0] - centroids_sorted_x[i-1, 0]
        if 1.0 < diff < 100:  # Filter out very small or large differences
            x_diffs.append(diff)
    
    # Find differences between consecutive y coordinates
    for i in range(1, len(centroids_sorted_y)):
        diff = centroids_sorted_y[i, 1] - centroids_sorted_y[i-1, 1]
        if 1.0 < diff < 100:  # Filter out very small or large differences
            y_diffs.append(diff)
    
    # Use clustering to find the most common differences
    if x_diffs:
        x_diffs = np.array(x_diffs).reshape(-1, 1)
        x_clustering = DBSCAN(eps=2.0, min_samples=2).fit(x_diffs)
        x_labels = x_clustering.labels_
        
        # Find the most common cluster
        if len(set(x_labels)) > 1:  # More than just noise
            most_common_cluster = max(set(x_labels) - {-1}, 
                                     key=lambda x: np.sum(x_labels == x))
            col_step = np.median(x_diffs[x_labels == most_common_cluster])
        else:
            col_step = np.median(x_diffs)
    else:
        # Fallback: use median block width
        widths = [block['dimensions'][0] for block in blocks]
        col_step = np.median(widths)
    
    if y_diffs:
        y_diffs = np.array(y_diffs).reshape(-1, 1)
        y_clustering = DBSCAN(eps=2.0, min_samples=2).fit(y_diffs)
        y_labels = y_clustering.labels_
        
        # Find the most common cluster
        if len(set(y_labels)) > 1:  # More than just noise
            most_common_cluster = max(set(y_labels) - {-1}, 
                                     key=lambda x: np.sum(y_labels == x))
            row_step = np.median(y_diffs[y_labels == most_common_cluster])
        else:
            row_step = np.median(y_diffs)
    else:
        # Fallback: use median block height
        heights = [block['dimensions'][1] for block in blocks]
        row_step = np.median(heights)
    
    # Calculate offsets (to align grid with blocks)
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 1]
    
    # Find the most common remainder when dividing by step size
    x_remainders = x_coords % col_step
    y_remainders = y_coords % row_step
    
    # Cluster the remainders to find the most common offset
    x_remainders = x_remainders.reshape(-1, 1)
    y_remainders = y_remainders.reshape(-1, 1)
    
    x_clustering = KMeans(n_clusters=min(3, len(x_remainders))).fit(x_remainders)
    y_clustering = KMeans(n_clusters=min(3, len(y_remainders))).fit(y_remainders)
    
    col_offset = x_clustering.cluster_centers_[np.argmax(np.bincount(x_clustering.labels_))][0]
    row_offset = y_clustering.cluster_centers_[np.argmax(np.bincount(y_clustering.labels_))][0]
    
    logging.info(f"Estimated grid: row_step={row_step:.2f}, col_step={col_step:.2f}, "
                f"row_offset={row_offset:.2f}, col_offset={col_offset:.2f}")
    
    # Debug visualization
    if debug_dir:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(debug_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        plt.hist(x_diffs, bins=20)
        plt.title(f'X Differences (step: {col_step:.2f})')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.hist(y_diffs, bins=20)
        plt.title(f'Y Differences (step: {row_step:.2f})')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.hist(x_remainders, bins=20)
        plt.title(f'X Remainders (offset: {col_offset:.2f})')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.hist(y_remainders, bins=20)
        plt.title(f'Y Remainders (offset: {row_offset:.2f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, 'grid_estimation.png'))
        plt.close()
    
    return (row_step, col_step, row_offset, col_offset)

def quantize_colors(
    blocks: List[Dict],
    num_colors: int = 16
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Quantize block colors to a smaller palette.
    
    Args:
        blocks: List of block feature dictionaries
        num_colors: Target number of colors
        
    Returns:
        Dictionary mapping block IDs to quantized colors
    """
    if not blocks:
        return {}
    
    # Extract colors and block IDs
    colors = []
    block_ids = []
    
    for block in blocks:
        color = block['color']
        if len(color) == 3:
            # Add alpha channel if missing
            color = color + (255,)
        colors.append(color)
        block_ids.append(block['id'])
    
    # Convert to numpy array
    colors_array = np.array(colors)
    
    # Separate RGB and alpha channels
    rgb = colors_array[:, :3]
    alpha = colors_array[:, 3]
    
    # Count unique colors
    unique_rgb = np.unique(rgb, axis=0)
    actual_num_colors = min(num_colors, len(unique_rgb))
    
    if actual_num_colors <= 1:
        # Only one color, no need for clustering
        quantized_colors = {block_id: tuple(colors_array[i]) for i, block_id in enumerate(block_ids)}
        return quantized_colors
    
    # Apply K-means clustering to RGB values
    kmeans = KMeans(n_clusters=actual_num_colors, random_state=0, n_init=10).fit(rgb)
    
    # Get cluster centers (the new palette)
    palette_rgb = kmeans.cluster_centers_.astype(np.uint8)
    
    # Map original colors to nearest cluster center
    labels = kmeans.predict(rgb)
    
    # Determine representative alpha for each cluster
    cluster_alphas = {}
    for i in range(actual_num_colors):
        cluster_mask = (labels == i)
        if np.any(cluster_mask):
            # Use median alpha
            median_alpha = np.median(alpha[cluster_mask]).astype(np.uint8)
            cluster_alphas[i] = median_alpha
        else:
            cluster_alphas[i] = 255  # Default to opaque
    
    # Create mapping from block ID to quantized color
    quantized_colors = {}
    for i, block_id in enumerate(block_ids):
        cluster_idx = labels[i]
        quantized_rgb = tuple(palette_rgb[cluster_idx])
        quantized_alpha = cluster_alphas[cluster_idx]
        quantized_colors[block_id] = quantized_rgb + (quantized_alpha,)
    
    logging.info(f"Quantized {len(colors)} colors to {actual_num_colors} colors")
    return quantized_colors

def snap_to_grid(
    blocks: List[Dict],
    row_step: float,
    col_step: float,
    row_offset: float,
    col_offset: float
) -> Dict[Tuple[int, int], List[int]]:
    """
    Snap block centroids to the nearest grid point.
    
    Args:
        blocks: List of block feature dictionaries
        row_step: Vertical distance between grid points
        col_step: Horizontal distance between grid points
        row_offset: Vertical offset of the grid
        col_offset: Horizontal offset of the grid
        
    Returns:
        Dictionary mapping grid coordinates to lists of block IDs
    """
    grid = defaultdict(list)
    
    for block in blocks:
        cx, cy = block['centroid']
        
        # Calculate nearest grid point
        grid_x = round((cx - col_offset) / col_step)
        grid_y = round((cy - row_offset) / row_step)
        
        # Add block to grid
        grid[(grid_y, grid_x)].append(block['id'])
    
    logging.info(f"Snapped {len(blocks)} blocks to {len(grid)} grid points")
    return grid

def resolve_grid_conflicts(
    grid: Dict[Tuple[int, int], List[int]],
    blocks: List[Dict],
    quantized_colors: Dict[int, Tuple[int, int, int, int]]
) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Resolve conflicts where multiple blocks map to the same grid point.
    
    Args:
        grid: Dictionary mapping grid coordinates to lists of block IDs
        blocks: List of block feature dictionaries
        quantized_colors: Dictionary mapping block IDs to quantized colors
        
    Returns:
        Dictionary mapping grid coordinates to final colors
    """
    resolved_grid = {}
    conflicts = 0
    
    for grid_pos, block_ids in grid.items():
        if len(block_ids) == 1:
            # No conflict
            block_id = block_ids[0]
            if block_id in quantized_colors:
                resolved_grid[grid_pos] = quantized_colors[block_id]
        else:
            # Conflict - use the largest block's color
            conflicts += 1
            block_areas = [(block_id, next((b['area'] for b in blocks if b['id'] == block_id), 0))
                          for block_id in block_ids]
            block_areas.sort(key=lambda x: x[1], reverse=True)
            
            # Use the largest block's color
            largest_block_id = block_areas[0][0]
            if largest_block_id in quantized_colors:
                resolved_grid[grid_pos] = quantized_colors[largest_block_id]
    
    if conflicts > 0:
        logging.info(f"Resolved {conflicts} grid conflicts")
    
    return resolved_grid

def create_pixel_art(
    grid: Dict[Tuple[int, int], Tuple[int, int, int, int]]
) -> Image.Image:
    """
    Create pixel art image from the grid.
    
    Args:
        grid: Dictionary mapping grid coordinates to colors
        
    Returns:
        PIL Image of the pixel art
    """
    if not grid:
        logging.error("Empty grid, cannot create pixel art")
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    
    # Find grid extents
    grid_positions = np.array(list(grid.keys()))
    min_y, min_x = np.min(grid_positions, axis=0)
    max_y, max_x = np.max(grid_positions, axis=0)
    
    # Calculate dimensions
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    
    # Create image
    img_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Fill in pixels
    for (grid_y, grid_x), color in grid.items():
        y = grid_y - min_y
        x = grid_x - min_x
        if 0 <= y < height and 0 <= x < width:
            img_array[y, x] = color
    
    logging.info(f"Created pixel art image of size {width}x{height}")
    return Image.fromarray(img_array, 'RGBA')

def process_image(
    image: Image.Image,
    num_colors: int = 16,
    debug_dir: Optional[str] = None,
    asset_idx: Optional[int] = None
) -> PixelPerfecterResult:
    """
    Process an image to convert implied pixels to actual pixel art.
    
    Args:
        image: PIL Image
        num_colors: Target number of colors
        debug_dir: Optional directory for debug output
        asset_idx: Asset index for debug output
        
    Returns:
        PixelPerfecterResult containing the processed image and metadata
    """
    logging.info("Starting pixel perfecter processing...")
    
    # Step 1: Detect implied pixels
    contours = detect_implied_pixels(image, debug_dir, asset_idx)
    if not contours:
        logging.error("No implied pixels detected")
        return PixelPerfecterResult(image, (1, 1), 1)
    
    # Step 2: Extract block features
    blocks = extract_block_features(image, contours)
    if not blocks:
        logging.error("Failed to extract block features")
        return PixelPerfecterResult(image, (1, 1), 1)
    
    # Step 3: Estimate grid
    row_step, col_step, row_offset, col_offset = estimate_grid(blocks, debug_dir)
    
    # Step 4: Quantize colors
    quantized_colors = quantize_colors(blocks, num_colors)
    
    # Step 5: Snap blocks to grid
    grid = snap_to_grid(blocks, row_step, col_step, row_offset, col_offset)
    
    # Step 6: Resolve grid conflicts
    resolved_grid = resolve_grid_conflicts(grid, blocks, quantized_colors)
    
    # Step 7: Create pixel art
    pixel_art = create_pixel_art(resolved_grid)
    
    # Calculate scale (average of row and column steps)
    scale = int(round((row_step + col_step) / 2))
    
    # Debug visualization
    if debug_dir:
        import os
        from PIL import Image, ImageDraw
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create debug image showing grid
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        
        # Draw grid
        img_width, img_height = image.size
        for x in range(int(col_offset), img_width, int(col_step)):
            draw.line([(x, 0), (x, img_height)], fill=(255, 0, 0, 128), width=1)
        for y in range(int(row_offset), img_height, int(row_step)):
            draw.line([(0, y), (img_width, y)], fill=(255, 0, 0, 128), width=1)
        
        # Draw block centroids
        for block in blocks:
            cx, cy = block['centroid']
            draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=(0, 255, 0, 128))
        
        prefix = f"asset_{asset_idx}_" if asset_idx is not None else ""
        debug_img.save(os.path.join(debug_dir, f"{prefix}grid_overlay.png"))
        
        # Save pixel art
        pixel_art.save(os.path.join(debug_dir, f"{prefix}pixel_art.png"))
    
    return PixelPerfecterResult(
        pixel_art,
        (pixel_art.height, pixel_art.width),
        scale
    )