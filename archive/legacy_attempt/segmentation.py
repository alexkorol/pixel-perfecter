# segmentation.py

import numpy as np
import cv2
import logging
from PIL import Image
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple, NamedTuple, Optional

class SegmentationResult(NamedTuple):
    labels: np.ndarray
    colors: Dict[int, Tuple[int, int, int, int]]

def segment_image(
image: Image.Image,
scale: int,
method: str = 'watershed',
asset_idx: Optional[int] = None,
debug_dir: Optional[str] = None,
blur_size: int = 5
) -> Optional[SegmentationResult]:
    """
    Segment the image into regions that correspond to pixel art pixels.
    
    Args:
        image: PIL Image
        scale: Estimated scale (used as a hint but not strictly enforced)
        method: Segmentation method ('watershed' or 'kmeans')
        asset_idx: Asset index for debug output
        debug_dir: Directory for debug output
        
    Returns:
        SegmentationResult with labels and colors, or None if segmentation failed
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Handle different image formats (RGB, RGBA, Grayscale)
    if image_np.ndim == 2:  # Grayscale
        logging.warning("Input image is grayscale, converting to RGB.")
        rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
    elif image_np.shape[2] == 3:  # RGB
        rgb = image_np
        alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
    elif image_np.shape[2] == 4:  # RGBA
        rgb = image_np[..., :3]
        alpha = image_np[..., 3]
    else:
        logging.error(f"Unsupported image shape: {image_np.shape}")
        return None

    # --- Create alpha mask with appropriate threshold ---
    # Use a lower threshold (10) to ensure we capture semi-transparent pixels
    # This is especially important when blur_size is 0
    alpha_mask = alpha > 10  # Mask out mostly transparent areas

    if method == 'watershed':
        # --- Preprocessing for Noise Reduction ---
        if blur_size > 0:
            logging.info(f"Applying Median Blur (k={blur_size}) for noise reduction...")
            # Apply Median Blur to the RGB image to reduce internal noise/dithering
            rgb_blurred = cv2.medianBlur(rgb, blur_size)
        else:
            # When blur_size is 0 (disabled), we still need some noise reduction for segmentation to work
            # Use an adaptive default blur size based on the image scale
            # For larger scales (more pixels per game pixel), we need more blurring
            
            # Log detailed information about the scale and image
            logging.info(f"Image dimensions: {rgb.shape}")
            logging.info(f"Detected scale: {scale}")
            
            # Calculate adaptive blur size
            adaptive_blur = min(max(scale // 2, 3), 9)
            if adaptive_blur % 2 == 0:  # Ensure it's odd
                adaptive_blur += 1
            
            logging.info(f"User disabled blur (k=0), but using adaptive blur (k={adaptive_blur}) for better segmentation")
            rgb_blurred = cv2.medianBlur(rgb, adaptive_blur)

        # --- Calculate Gradient ---
        logging.info("Calculating gradient magnitude...")
        # Calculate gradient on the blurred image
        gradient = np.zeros_like(rgb_blurred, dtype=float)
        for i in range(3):
            # Use Sobel operator for gradient calculation
            gradient[..., i] = ndimage.sobel(rgb_blurred[..., i].astype(float))
        # Calculate magnitude across color channels
        gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=2))

        # --- Optional: Slight blur on gradient map ---
        # gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (3, 3), 0)

        # --- Generate Markers using Distance Transform and Local Maxima ---
        logging.info("Generating markers using distance transform and local maxima...")

        # Create binary mask from alpha channel
        # Ensure binary_mask is uint8 for connectedComponents
        binary_mask = (alpha_mask > 0).astype(np.uint8)
        
        # Apply distance transform to binary mask
        dist_transform = distance_transform_edt(binary_mask)
        
        # Find local maxima in the distance transform
        # This will create more markers for watershed
        local_max_footprint = np.ones((3, 3))
        local_max = peak_local_max(
            dist_transform,
            min_distance=max(3, scale//3),  # Adjust based on scale
            footprint=local_max_footprint,
            labels=binary_mask
        )
        
        # Create markers from local maxima
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        
        # As a fallback, also use connected components
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        logging.info(f"Found {num_labels - 1} potential markers via connected components (excluding background).")

        # Count existing markers from local maxima
        existing_marker_count = len(np.unique(markers)) - 1  # Subtract 1 for background
        logging.info(f"Created {existing_marker_count} markers from local maxima.")
        
        # Add markers from connected components for better coverage
        cc_marker_count = 0
        min_component_size = max(5, scale*scale//16)  # Scale-dependent minimum size
        
        for i in range(1, num_labels):  # Skip background label 0
            centroid_x, centroid_y = centroids[i]
            component_size = stats[i, cv2.CC_STAT_AREA]
            
            # Only add large components that don't already have a marker nearby
            if component_size >= min_component_size:
                r, c = int(centroid_y), int(centroid_x)
                
                # Check if there's already a marker nearby
                if 0 <= r < markers.shape[0] and 0 <= c < markers.shape[1]:
                    # Check 5x5 neighborhood
                    r_start = max(0, r-2)
                    r_end = min(markers.shape[0], r+3)
                    c_start = max(0, c-2)
                    c_end = min(markers.shape[1], c+3)
                    neighborhood = markers[r_start:r_end, c_start:c_end]
                    
                    # If no markers in neighborhood, add one
                    if np.max(neighborhood) == 0:
                        cc_marker_count += 1
                        markers[r, c] = existing_marker_count + cc_marker_count


        total_marker_count = existing_marker_count + cc_marker_count
        logging.info(f"Created marker image with {total_marker_count} total markers ({existing_marker_count} from local maxima, {cc_marker_count} from components).")
        
        if total_marker_count == 0:
            logging.warning("No markers were placed. Segmentation might fail.")

        # --- Apply Watershed ---
        logging.info("Applying watershed algorithm...")
        # Run watershed on the original gradient magnitude using the generated markers
        labels = watershed(gradient_magnitude, markers, mask=alpha_mask)
        logging.info(f"Watershed segmentation complete. Found {labels.max()} segments.")

    elif method == 'kmeans':
        # Kmeans implementation needs significant revision for irregular grids
        # For now, raise error or return None
        logging.error("KMeans method is not currently supported well for irregular grids.")
        # You might reuse parts of your old kmeans code here if you adapt it later
        return None  # Or implement adaptive Kmeans/SLIC if needed
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    # --- Calculate Median Segment Colors ---
    logging.info("Calculating segment colors...")
    colors_dict = {}
    unique_segment_labels = np.unique(labels)
    # Create RGBA array for median calculation
    image_rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    image_rgba[..., :3] = rgb
    image_rgba[..., 3] = alpha  # Ensure correct alpha is used

    for label_id in unique_segment_labels:
        if label_id <= 0:  # Skip background (0) or watershed boundaries (-1)
            continue
        mask = (labels == label_id)
        if not np.any(mask):
            continue

        # Calculate median color (RGBA) for the segment
        segment_pixels = image_rgba[mask]
        if len(segment_pixels) > 0:
            median_color = np.median(segment_pixels, axis=0).astype(np.uint8)
            # Ensure alpha is reasonable (if median is 0, but should be opaque)
            if median_color[3] < 128 and np.median(alpha[mask]) >= 128:
                median_color[3] = 255  # Make fully opaque if mostly opaque
            colors_dict[label_id] = tuple(median_color)  # Store as tuple

    logging.info(f"Calculated colors for {len(colors_dict)} segments.")

    # --- Debug Visualizations ---
    if debug_dir:
        import os
        from PIL import Image
        os.makedirs(debug_dir, exist_ok=True)  # Ensure dir exists
        prefix = f"asset_{asset_idx}_" if asset_idx is not None else ""

        # Save blurred RGB (always save if watershed method)
        if method == 'watershed' and 'rgb_blurred' in locals():
            Image.fromarray(rgb_blurred).save(os.path.join(debug_dir, f"{prefix}dbg_blurred_rgb.png"))
        # Save gradient magnitude (always save if watershed method)
        if method == 'watershed' and 'gradient_magnitude' in locals():
            grad_vis = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            Image.fromarray(grad_vis).save(os.path.join(debug_dir, f"{prefix}dbg_gradient_magnitude.png"))
        # Save distance transform (always save if watershed method)
        if method == 'watershed' and 'dist_transform' in locals():
            dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            Image.fromarray(dist_vis).save(os.path.join(debug_dir, f"{prefix}dbg_distance_transform.png"))
        # Save markers visualization (always save if watershed method, even if empty)
        if method == 'watershed' and 'markers' in locals():
            marker_vis = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
            marker_vis[markers > 0] = [0, 255, 0]  # Green markers
            Image.fromarray(marker_vis).save(os.path.join(debug_dir, f"{prefix}dbg_markers.png"))
        # Save colored label map (only if labels exist and have segments > 0)
        if 'labels' in locals() and labels.max() > 0:
            vis_labels = np.unique(labels)
            # Filter out background label 0 if present for color mapping
            vis_labels_no_bg = vis_labels[vis_labels > 0]
            if len(vis_labels_no_bg) > 0:
                vis_colors = np.random.randint(0, 255, (len(vis_labels_no_bg), 3), dtype=np.uint8)
                label_vis_color = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) # Background black
                for i, label_id in enumerate(vis_labels_no_bg):
                    label_vis_color[labels == label_id] = vis_colors[i]
                Image.fromarray(label_vis_color).save(os.path.join(debug_dir, f"{prefix}dbg_segments_color.png"))
            else:
                logging.info("Skipping saving dbg_segments_color.png as no segments > 0 were found.")
        elif 'labels' in locals():
             logging.info("Skipping saving dbg_segments_color.png as labels array likely only contains background.")

        logging.info(f"Saved debug images to {debug_dir}")

    return SegmentationResult(labels, colors_dict)