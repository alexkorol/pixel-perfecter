import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import logging
from typing import Dict, Tuple, Optional, List, Set
from scipy import ndimage
import cv2
from sklearn.cluster import KMeans

# Import matplotlib with Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_segment_properties(
    label_map: np.ndarray,
    min_segment_size: int = 1 # Lowered threshold to keep even single-pixel segments
) -> Dict[int, Dict]:
    """
    Calculate properties for each segment including centroid, bounding box, and size.
    
    Args:
        label_map: 2D array of segment labels
        min_segment_size: Minimum number of pixels for a valid segment
        
    Returns:
        Dictionary mapping segment IDs to their properties
    """
    logging.info("Calculating segment properties...")
    unique_labels = np.unique(label_map)
    segment_properties = {}
    
    # Skip background/watershed (-1 or 0)
    for label in unique_labels:
        if label <= 0:
            continue
            
        # Create mask for this segment
        mask = (label_map == label)
        size = np.sum(mask)
        
        # Skip very small segments
        if size < min_segment_size:
            logging.debug(f"Skipping segment {label} with only {size} pixels")
            continue
            
        # Get centroid (center of mass)
        cy, cx = ndimage.center_of_mass(mask)
        
        # Get bounding box (min_row, min_col, max_row, max_col)
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:
            continue
            
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        # Calculate dimensions
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        segment_properties[label] = {
            'centroid': (cy, cx),
            'bbox': (min_row, min_col, max_row, max_col),
            'dimensions': (height, width),
            'size': size
        }
    
    logging.info(f"Found properties for {len(segment_properties)} valid segments")
    return segment_properties

def estimate_quantization_steps(
    segment_properties: Dict[int, Dict],
    debug_dir: Optional[str] = None
) -> Tuple[float, float]:
    """
    Estimate the quantization steps for row and column coordinates.
    
    Args:
        segment_properties: Dictionary of segment properties
        debug_dir: Optional directory for debug output
        
    Returns:
        Tuple of (row_step, col_step) in pixels
    """
    logging.info("Estimating quantization steps for segment positioning...")
    
    # Get all centroids
    centroids = [prop['centroid'] for prop in segment_properties.values()]
    if not centroids:
        logging.error("No valid centroids found")
        return (8.0, 8.0)  # Fallback to default step
        
    centroids = np.array(centroids)
    
    # Calculate the minimum non-zero distances between adjacent centroids
    row_diffs = []
    col_diffs = []
    
    # Sort centroids by row then column
    centroids_sorted_row = centroids[np.argsort(centroids[:, 0])]
    
    # Find differences between consecutive rows
    for i in range(1, len(centroids_sorted_row)):
        diff = centroids_sorted_row[i, 0] - centroids_sorted_row[i-1, 0]
        if 1.0 < diff < 100:  # Skip tiny or huge differences
            row_diffs.append(diff)
            
    # Sort centroids by column then row
    centroids_sorted_col = centroids[np.argsort(centroids[:, 1])]
    
    # Find differences between consecutive columns
    for i in range(1, len(centroids_sorted_col)):
        diff = centroids_sorted_col[i, 1] - centroids_sorted_col[i-1, 1]
        if 1.0 < diff < 100:  # Skip tiny or huge differences
            col_diffs.append(diff)
    
    # Get median differences to estimate steps
    if row_diffs:
        row_step = np.median(row_diffs)
    else:
        # Fallback: use median segment height
        heights = [prop['dimensions'][0] for prop in segment_properties.values()]
        row_step = max(np.median(heights), 5.0)
        
    if col_diffs:
        col_step = np.median(col_diffs)
    else:
        # Fallback: use median segment width
        widths = [prop['dimensions'][1] for prop in segment_properties.values()]
        col_step = max(np.median(widths), 5.0)
    
    logging.info(f"Estimated quantization steps: row={row_step:.2f}, col={col_step:.2f}")
    
    # Debug: Create histogram of differences
    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.hist(row_diffs, bins=20)
        plt.title(f'Row Diffs (median: {row_step:.2f})')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.hist(col_diffs, bins=20)
        plt.title(f'Column Diffs (median: {col_step:.2f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, 'quantization_steps_hist.png'))
        plt.close()
    
    return (row_step, col_step)

def reconstruct_image_adaptive(
    label_map: np.ndarray,
    segment_colors: Dict[int, Tuple[int, int, int, int]],
    num_colors: int = 16,  # Add parameter for target palette size
    debug_dir: Optional[str] = None
) -> Image.Image:
    """
    Reconstructs the pixel art by analyzing segments and their positions.
    
    Args:
        label_map: 2D array of segment labels
        segment_colors: Dict mapping segment IDs to RGBA colors
        debug_dir: Optional directory for debug output
    
    Returns:
        PIL Image containing the reconstructed pixel art
    """
    logging.info("Starting adaptive reconstruction...")
    
    # Calculate segment properties
    segment_props = calculate_segment_properties(label_map)
    logging.info(f"Calculated properties for {len(segment_props)} segments.")
    if not segment_props:
        logging.warning("No segment properties calculated (all segments might be too small).")
        # Early exit if no properties, as quantization steps and centroids will fail
        raise ValueError("No valid segments found after property calculation (check min_segment_size or segmentation result).")

    # Estimate quantization steps
    row_step, col_step = estimate_quantization_steps(segment_props, debug_dir)

    # --- Quantize Segment Colors ---
    logging.info(f"Quantizing {len(segment_colors)} segment colors to {num_colors} colors...")
    quantized_segment_colors = quantize_colors(segment_colors, num_colors)
    logging.info(f"Quantization resulted in {len(quantized_segment_colors)} quantized colors mapped back to original labels.")
    logging.info(f"Color quantization complete. Mapped to {len(np.unique([v for v in quantized_segment_colors.values()], axis=0))} unique colors.")

    # Create a grid to snap centroids
    centroids_quantized = {}
    for label, props in segment_props.items():
        # Use quantized colors
        if label not in quantized_segment_colors:
            continue
            
        cy, cx = props['centroid']
        # Quantize the centroid coordinates to multiples of the steps
        qy = round(cy / row_step)
        qx = round(cx / col_step)
        centroids_quantized[label] = (qy, qx)
    logging.info(f"Populated {len(centroids_quantized)} quantized centroids.")

    # Check if any centroids were quantized
    if not centroids_quantized:
        raise ValueError("No valid segments found to reconstruct after processing and quantization. Check input image or segmentation parameters.")

    # Find grid extents for output image
    min_qy = min(qy for qy, _ in centroids_quantized.values())
    max_qy = max(qy for qy, _ in centroids_quantized.values())
    min_qx = min(qx for _, qx in centroids_quantized.values())
    max_qx = max(qx for _, qx in centroids_quantized.values())
    
    # Calculate output image dimensions
    h_out = max_qy - min_qy + 1
    w_out = max_qx - min_qx + 1
    logging.info(f"Output dimensions: {w_out}x{h_out} pixels")
    
    # Create output image
    output_array = np.zeros((h_out, w_out, 4), dtype=np.uint8)
    
    # Place colors in output image
    placed_pixels = 0
    for label, (qy, qx) in centroids_quantized.items():
        # Use quantized colors
        if label not in quantized_segment_colors:
            continue
            
        # Calculate normalized position in output
        y_out = qy - min_qy
        x_out = qx - min_qx
        
        # Ensure we're within bounds
        if 0 <= y_out < h_out and 0 <= x_out < w_out:
            # Use quantized colors
            output_array[y_out, x_out] = quantized_segment_colors[label]
            placed_pixels += 1
            
    logging.info(f"Placed {placed_pixels} pixels in output")
    
    # Create conflict resolution map for the output image
    grid_cells = {}
    for label, (qy, qx) in centroids_quantized.items():
        y_out = qy - min_qy
        x_out = qx - min_qx
        
        if 0 <= y_out < h_out and 0 <= x_out < w_out:
            cell_key = (y_out, x_out)
            if cell_key not in grid_cells:
                grid_cells[cell_key] = []
            grid_cells[cell_key].append((label, segment_props[label]['size']))
            
    # Resolve conflicts using segment size
    conflicts_resolved = 0
    for (y, x), segments in grid_cells.items():
        if len(segments) > 1:
            conflicts_resolved += 1
            # Sort by segment size (largest first)
            segments.sort(key=lambda s: s[1], reverse=True)
            # Use the largest segment's quantized color
            winner_label = segments[0][0]
            if winner_label in quantized_segment_colors:
                 output_array[y, x] = quantized_segment_colors[winner_label]
            else:
                 # Fallback if somehow the winner label wasn't in the quantized dict
                 # This shouldn't happen if quantization included all original labels
                 logging.warning(f"Winner label {winner_label} not found in quantized colors during conflict resolution.")
                 # Find the first available color from the conflicting segments
                 for lbl, _ in segments:
                     if lbl in quantized_segment_colors:
                         output_array[y, x] = quantized_segment_colors[lbl]
                         break
    
    if conflicts_resolved > 0:
        logging.info(f"Resolved {conflicts_resolved} cell conflicts using segment size")
    
    # Fill any remaining transparent gaps
    output_array = fill_transparent_gaps(output_array)
    
    # Save a debug image showing the quantized centroids
    if debug_dir:
        import os
        from matplotlib.patches import Rectangle
        
        # Original image with segment centers
        plt.figure(figsize=(10, 10))
        
        # Create a visualization of the original segments
        vis_labels = np.unique(label_map)
        vis_colors = np.random.randint(0, 255, (len(vis_labels), 3), dtype=np.uint8)
        vis_colors[0] = [0, 0, 0]  # Background black
        
        label_vis_color = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
        for i, label_id in enumerate(vis_labels):
            label_vis_color[label_map == label_id] = vis_colors[i]
            
        plt.imshow(label_vis_color)
        
        # Draw original centroids (blue) and quantized ones (red)
        for label, props in segment_props.items():
            if label in centroids_quantized:
                cy, cx = props['centroid']
                qy, qx = centroids_quantized[label]
                
                # Draw original centroid in blue
                plt.plot(cx, cy, 'bo', markersize=3)
                
                # Draw quantized centroid in red
                plt.plot(qx * col_step, qy * row_step, 'ro', markersize=3)
                
                # Draw line connecting them
                plt.plot([cx, qx * col_step], [cy, qy * row_step], 'g-', linewidth=0.5)
        
        plt.title('Segment Centroids: Original (blue) and Quantized (red)')
        plt.savefig(os.path.join(debug_dir, 'centroids_quantized.png'))
        plt.close()
        
    return Image.fromarray(output_array, 'RGBA')


def quantize_colors(
    segment_colors: Dict[int, Tuple[int, int, int, int]],
    num_colors: int
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Reduces the number of colors in the segment dictionary using K-Means clustering.

    Args:
        segment_colors: Dictionary mapping segment IDs to RGBA colors.
        num_colors: The target number of colors.

    Returns:
        A new dictionary mapping segment IDs to the closest quantized RGBA color.
    """
    if not segment_colors:
        return {}

    original_labels = list(segment_colors.keys())
    # Extract colors (RGB only for clustering, handle alpha separately)
    colors_rgb = np.array([c[:3] for c in segment_colors.values()], dtype=np.float32)
    alphas = np.array([c[3] for c in segment_colors.values()], dtype=np.uint8)

    # Handle case where we have fewer unique colors than requested clusters
    unique_colors_rgb = np.unique(colors_rgb, axis=0)
    actual_num_colors = min(num_colors, len(unique_colors_rgb))

    if actual_num_colors <= 0: # No colors to quantize
        return {}
        
    if actual_num_colors == 1: # Only one color, no need for kmeans
        center_rgb = unique_colors_rgb[0]
        # Find the most common alpha for this color
        dominant_alpha = np.median(alphas[np.all(colors_rgb == center_rgb, axis=1)]).astype(np.uint8)
        quantized_color_rgba = tuple(center_rgb.astype(np.uint8)) + (dominant_alpha,)
        return {label: quantized_color_rgba for label in original_labels}


    logging.info(f"Running K-Means with k={actual_num_colors} on {len(colors_rgb)} colors...")
    kmeans = KMeans(n_clusters=actual_num_colors, random_state=0, n_init=10).fit(colors_rgb)
    
    # Get the cluster centers (the new palette)
    palette_rgb = kmeans.cluster_centers_.astype(np.uint8)

    # Map original colors to the nearest cluster center
    predicted_labels = kmeans.predict(colors_rgb)

    # Create the new color dictionary
    quantized_segment_colors = {}
    
    # Determine the representative alpha for each cluster
    cluster_alphas = {}
    for i in range(actual_num_colors):
        cluster_mask = (predicted_labels == i)
        if np.any(cluster_mask):
            # Use median alpha of the original colors belonging to this cluster
            median_alpha = np.median(alphas[cluster_mask]).astype(np.uint8)
            # Ensure reasonable opacity if original pixels were mostly opaque
            if median_alpha < 128 and np.median(alphas[cluster_mask]) >= 128:
                 median_alpha = 255
            cluster_alphas[i] = median_alpha
        else:
            cluster_alphas[i] = 255 # Default to opaque if cluster is empty (shouldn't happen)


    for i, original_label in enumerate(original_labels):
        cluster_index = predicted_labels[i]
        quantized_rgb = tuple(palette_rgb[cluster_index])
        quantized_alpha = cluster_alphas[cluster_index]
        quantized_segment_colors[original_label] = quantized_rgb + (quantized_alpha,)

    return quantized_segment_colors

def fill_transparent_gaps(img_array: np.ndarray, max_iterations=5) -> np.ndarray:
    """
    Fill transparent gaps using nearest neighbor colors. Iteratively fills pixels
    adjacent to already colored pixels.

    Args:
        img_array: RGBA numpy array.
        max_iterations: Maximum number of passes to fill gaps.

    Returns:
        RGBA numpy array with gaps filled using nearest neighbors.
    """
    h, w = img_array.shape[:2]
    filled = img_array.copy()
    filled_count = 0
    
    filled = img_array.copy()
    filled_count_total = 0

    for iteration in range(max_iterations):
        filled_this_iteration = 0
        # Find transparent pixels *in the current state*
        transparent_mask = filled[..., 3] == 0
        
        if not np.any(transparent_mask):
            logging.info(f"No more transparent pixels found after iteration {iteration}.")
            break # Stop if no gaps left

        # Get coordinates of transparent pixels
        transparent_coords = np.argwhere(transparent_mask)
        
        # Create a copy to update, avoid modifying while iterating
        updated_filled = filled.copy()

        for y, x in transparent_coords:
            # Find the closest opaque neighbor (Manhattan distance for simplicity)
            best_neighbor_color = None
            min_dist = float('inf')

            # Search outwards in increasing distance (limited search radius for efficiency)
            search_radius = 3 # Look up to 3 pixels away
            found_neighbor = False
            for dist in range(1, search_radius + 1):
                for dy in range(-dist, dist + 1):
                    for dx in range(-dist, dist + 1):
                         # Only check perimeter of the square box
                        if abs(dy) != dist and abs(dx) != dist:
                            continue

                        ny, nx = y + dy, x + dx

                        if 0 <= ny < h and 0 <= nx < w:
                            # Check the *original* filled array for opaque pixels
                            if filled[ny, nx, 3] > 0:
                                current_dist = abs(dy) + abs(dx) # Manhattan distance
                                if current_dist < min_dist:
                                     min_dist = current_dist
                                     best_neighbor_color = filled[ny, nx]
                                     found_neighbor = True
                
                if found_neighbor: # Found closest neighbor in this radius
                    break
            
            # If a neighbor was found, fill the pixel in the updated array
            if best_neighbor_color is not None:
                updated_filled[y, x] = best_neighbor_color
                filled_this_iteration += 1
        
        # Update the main filled array for the next iteration
        filled = updated_filled
        filled_count_total += filled_this_iteration
        logging.debug(f"Gap fill iteration {iteration + 1}: Filled {filled_this_iteration} pixels.")

        if filled_this_iteration == 0:
             logging.info(f"No pixels filled in iteration {iteration + 1}, stopping.")
             break # Stop if no progress is made
    
    if filled_count_total > 0:
        logging.info(f"Filled {filled_count_total} transparent pixels in total using nearest neighbor.")
        
    return filled