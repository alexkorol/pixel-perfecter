import cv2
import numpy as np
from collections import Counter
from typing import Optional, List, Tuple
import logging

def detect_lines(edges: np.ndarray, min_line_length: int = 10, max_line_gap: int = 5) -> Optional[np.ndarray]:
    """
    Detect line segments using probabilistic Hough transform.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    return lines

def classify_line_angle(x1: int, y1: int, x2: int, y2: int, angle_threshold: float = 15) -> str:
    """
    Classify a line as horizontal, vertical, or diagonal based on its angle.
    """
    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
    
    if angle < angle_threshold or abs(angle - 180) < angle_threshold:
        return 'horizontal'
    elif abs(angle - 90) < angle_threshold:
        return 'vertical'
    return 'diagonal'

def measure_feature_thickness(
    edges: np.ndarray,
    gray: np.ndarray,
    x: int,
    y: int,
    direction: str,
    max_scan: int = 30
) -> Optional[int]:
    """
    Measure the thickness of a feature by scanning perpendicular to its direction.
    """
    h, w = edges.shape
    thickness = 0
    in_feature = False
    
    if direction == 'horizontal':
        # Scan vertically
        for dy in range(-max_scan, max_scan + 1):
            ny = y + dy
            if 0 <= ny < h and edges[ny, x] > 0:
                thickness += 1
                in_feature = True
            elif in_feature:
                break
    else:  # vertical
        # Scan horizontally
        for dx in range(-max_scan, max_scan + 1):
            nx = x + dx
            if 0 <= nx < w and edges[y, nx] > 0:
                thickness += 1
                in_feature = True
            elif in_feature:
                break
                
    return thickness if thickness >= 4 else None

def estimate_scale_from_features(
    image,
    canny_low: int = 30,  # Lower threshold for more edge sensitivity
    canny_high: int = 90,  # Reduced high threshold
    debug_dir: Optional[str] = None,
    asset_idx: Optional[int] = None
) -> Optional[int]:
    """
    Estimate pixel art scale by measuring thin horizontal/vertical features.
    """
    # Convert to numpy arrays
    img_np = np.array(image)
    if img_np.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
        alpha = img_np[..., 3]
        # Create alpha mask
        _, alpha_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:  # RGB
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        alpha_mask = np.ones_like(gray) * 255
    
    # Apply light Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    
    # Edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.bitwise_and(edges, edges, mask=alpha_mask)
    
    if debug_dir:
        import os
        from PIL import Image
        prefix = f"asset_{asset_idx}_" if asset_idx is not None else ""
        debug_path = os.path.join(debug_dir, f"{prefix}edges.png")
        Image.fromarray(edges).save(debug_path)
    
    # Detect lines with optimal parameters for scale 8-16
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,  # Increased threshold for stronger lines
        minLineLength=12,  # Increased to better match expected feature size
        maxLineGap=4  # Slightly increased gap tolerance
    )
    
    if lines is None:
        logging.warning("No lines detected")
        return None
    
    # Measure feature thicknesses with more points per line
    thickness_estimates: List[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        direction = classify_line_angle(x1, y1, x2, y2, angle_threshold=20)  # More lenient angle
        
        if direction == 'diagonal':
            continue
            
        # Check thickness at more points along the line
        for t in np.linspace(0, 1, 7):  # Increased from 5 to 7 points
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            thickness = measure_feature_thickness(edges, gray, x, y, direction, max_scan=40)
            if thickness:
                thickness_estimates.append(thickness)
    
    if not thickness_estimates:
        logging.warning("No valid thickness measurements found")
        return None
    
    # Use robust statistics
    thickness_estimates.sort()
    # Remove outliers: keep values between 10th and 90th percentiles
    n = len(thickness_estimates)
    start_idx = int(n * 0.1)
    end_idx = int(n * 0.9)
    filtered_estimates = thickness_estimates[start_idx:end_idx]
    
    if not filtered_estimates:
        return None
        
    median_scale = int(round(np.median(filtered_estimates)))
    
    # Adjusted validation bounds
    if not (8 <= median_scale <= 50):  # Adjusted minimum bound to match typical pixel art
        logging.warning(f"Estimated scale {median_scale} outside reasonable bounds")
        return None
        
    logging.info(f"Thickness estimates: {thickness_estimates}")
    logging.info(f"Filtered estimates: {filtered_estimates}")
    logging.info(f"Median scale from features: {median_scale}")
    
    return median_scale

def detect_scale(image, debug_dir: Optional[str] = None) -> Optional[int]:
    """
    Detect the pixel art scale using multiple methods and combine their results.
    
    Args:
        image: PIL Image to analyze
        debug_dir: Optional directory for debug output
        
    Returns:
        int: Estimated scale factor, or None if detection fails
    """
    # Convert PIL Image to numpy arrays for processing
    img_np = np.array(image)
    if img_np.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
        alpha = img_np[..., 3]
        # Create alpha mask
        _, alpha_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:  # RGB
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        alpha_mask = np.ones_like(gray) * 255

    # Try feature-based detection first
    feature_scale = estimate_scale_from_features(image, debug_dir=debug_dir)
    
    if feature_scale and 8 <= feature_scale <= 32:
        logging.info(f"Selected scale {feature_scale} from feature detection")
        return feature_scale
        
    # If feature detection failed or gave unlikely results, use a default
    logging.warning("Scale detection failed or gave unlikely results")
    return None