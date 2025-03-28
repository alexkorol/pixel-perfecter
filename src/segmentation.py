# segmentation.py

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os

def create_colormap(n_colors):
    """Create a colormap for visualizing segments"""
    np.random.seed(42)  # For reproducible colors
    return np.random.randint(0, 255, (n_colors, 3), dtype=np.uint8)

def visualize_segments(label_map):
    """Convert a label map to a colored visualization"""
    unique_labels = np.unique(label_map)
    colormap = create_colormap(len(unique_labels))
    
    # Create visualization array
    vis = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    
    # Map each label to a color
    for i, label in enumerate(unique_labels):
        mask = label_map == label
        vis[mask] = colormap[i]
    
    return vis

def segment_image(image, scale, method='watershed', asset_idx=None, debug_dir=None):
    """
    Segments the high-resolution image into implied pixel regions using watershed or kmeans.
    Returns a label map with the same dimensions as the input image.
    """
    image_np = np.array(image)
    
    if method == 'watershed':
        # Convert to grayscale while preserving alpha
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        alpha = image_np[..., 3]
        
        # Create binary mask from alpha channel with more aggressive threshold
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        
        # Create grid-aligned markers based on scale
        h, w = image_np.shape[:2]
        grid_h = h // scale
        grid_w = w // scale
        
        # Create initial markers based on grid
        markers = np.zeros((h, w), dtype=np.int32)
        current_marker = 1
        
        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate grid cell boundaries
                y_start = i * scale
                y_end = min((i + 1) * scale, h)
                x_start = j * scale
                x_end = min((j + 1) * scale, w)
                
                # Find center of grid cell
                center_y = (y_start + y_end) // 2
                center_x = (x_start + x_end) // 2
                
                # Only create marker if pixel is non-transparent
                if alpha[center_y, center_x] > 127:
                    markers[center_y-1:center_y+2, center_x-1:center_x+2] = current_marker
                    current_marker += 1
        
        # Save debug images if directory is provided
        if debug_dir:
            Image.fromarray(gray).save(os.path.join(debug_dir, "01_gray.png"))
            Image.fromarray(binary).save(os.path.join(debug_dir, "02_binary.png"))
            
            # Visualize initial grid-based markers
            marker_vis = visualize_segments(markers)
            Image.fromarray(marker_vis).save(os.path.join(debug_dir, "03_initial_grid_markers.png"))
        
        # Apply watershed with the grid-based markers
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        cv2.watershed(image_bgr, markers)
        
        # Clean up markers by removing boundary markers (-1)
        markers[markers == -1] = 0
        
        # Log the number of segments found
        num_segments = len(np.unique(markers)) - 1  # Subtract 1 for background (0)
        print(f"Found {num_segments} segments")
        
        if debug_dir:
            # Save the final segmentation visualization
            segment_vis = visualize_segments(markers)
            Image.fromarray(segment_vis).save(os.path.join(debug_dir, "04_final_segments.png"))
        
        return markers
        
    elif method == 'kmeans':
        # Prepare feature vector combining color and position
        h, w = image_np.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        
        # Stack features: R,G,B,A, normalized X,Y position
        features = np.stack([
            image_np[..., 0].ravel(),
            image_np[..., 1].ravel(),
            image_np[..., 2].ravel(),
            image_np[..., 3].ravel(),
            x.ravel() / w * 255,  # Scale positions to similar range as colors
            y.ravel() / h * 255
        ], axis=1)
        
        # Only process non-transparent pixels
        mask = features[:, 3] > 0
        valid_features = features[mask]
        
        if len(valid_features) == 0:
            print("Warning: No non-transparent pixels found")
            return np.zeros_like(image_np[..., 0])
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(valid_features)
        
        # Estimate number of clusters based on target resolution
        n_clusters = max(int((h * w) / (scale * scale)), 4)  # At least 4 clusters
        print(f"Using {n_clusters} clusters for k-means")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        # Create full label map
        label_map = np.zeros(h * w, dtype=int)
        label_map[mask] = labels + 1  # Shift labels to reserve 0 for background
        label_map = label_map.reshape(h, w)
        
        if debug_dir:
            # Save the segmentation visualization
            segment_vis = visualize_segments(label_map)
            Image.fromarray(segment_vis).save(os.path.join(debug_dir, "kmeans_segments.png"))
        
        return label_map
    
    else:
        raise ValueError("Unsupported segmentation method.")

def calculate_segment_colors(label_map, high_res_image, method='median'):
    """
    Calculate representative colors for each segment in the label map.
    """
    image_np = np.array(high_res_image)
    unique_segments = np.unique(label_map)
    segment_colors = {}
    
    for segment in unique_segments:
        if segment <= 0:  # Skip background/boundaries
            continue
        
        mask = (label_map == segment)
        if not np.any(mask):  # Skip empty segments
            continue
            
        pixels = image_np[mask]
        if len(pixels) == 0:
            continue
            
        # Calculate color for all channels (RGBA)
        if method == 'median':
            color = np.median(pixels, axis=0)
        elif method == 'mean':
            color = np.mean(pixels, axis=0)
        else:
            raise ValueError("Unsupported color calculation method.")
        
        # Ensure proper range and type for RGBA values
        color = np.clip(np.round(color), 0, 255).astype(np.uint8)
        segment_colors[segment] = tuple(color)
    
    print(f"Calculated colors for {len(segment_colors)} segments")
    for segment_id, color in list(segment_colors.items())[:5]:
        print(f"Sample color for segment {segment_id}: {color}")
    return segment_colors

def calculate_segment_positions(label_map, scale):
    """
    Calculate the target pixel grid position for each segment.
    """
    unique_segments = np.unique(label_map)
    segment_positions = {}
    h, w = label_map.shape
    
    # Calculate grid size
    grid_h = h // scale
    grid_w = w // scale
    
    for segment in unique_segments:
        if segment <= 0:  # Skip background/boundaries
            continue
            
        coords = np.argwhere(label_map == segment)
        if len(coords) == 0:
            continue
            
        # Calculate centroid
        centroid_y, centroid_x = np.mean(coords, axis=0)
        
        # Map to target grid more precisely
        target_col = int(np.floor(centroid_x / scale))
        target_row = int(np.floor(centroid_y / scale))
        
        # Ensure within bounds
        target_col = min(max(0, target_col), grid_w - 1)
        target_row = min(max(0, target_row), grid_h - 1)
        
        segment_positions[segment] = (target_col, target_row)
        
        # Debug output for first few segments
        if len(segment_positions) <= 5:
            print(f"Segment {segment} centroid ({centroid_y:.1f}, {centroid_x:.1f}) -> position ({target_row}, {target_col})")
    
    print(f"Calculated positions for {len(segment_positions)} segments")
    return segment_positions