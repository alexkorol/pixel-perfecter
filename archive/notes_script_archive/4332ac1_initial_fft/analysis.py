import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import os

def find_harmonic_groups(scale_estimates, tolerance=0.12):
    """
    Group scale estimates that form harmonic series and find the most likely fundamental.
    """
    if len(scale_estimates) < 4:
        return None
        
    # Sort scales and convert to numpy array
    scales = np.array(sorted(scale_estimates))
    
    # Try each scale as a potential fundamental
    best_score = 0
    best_fundamental = None
    best_harmonics = None
    
    # Try smaller scales first (prefer finding the fundamental rather than harmonics)
    potential_fundamentals = np.concatenate([
        scales[(scales >= 8) & (scales <= 16)],  # Likely fundamentals
        scales[scales > 16] / 2,  # Potential half-harmonics
        scales[scales > 24] / 3   # Potential third harmonics
    ])
    
    for fundamental in np.unique(potential_fundamentals):
        # Look for harmonics (1x, 2x, 3x)
        harmonics = fundamental * np.array([1, 2, 3])
        matches = 0
        matched_scales = []
        
        # Count how many actual scales match these harmonics
        for harmonic in harmonics:
            # Find any scales that match this harmonic
            for scale in scales:
                error = abs(scale - harmonic) / harmonic
                if error < tolerance:
                    matches += 1
                    matched_scales.append(scale)
                    break
        
        # Score this fundamental based on:
        # 1. Number of harmonics found
        # 2. Preference for values around 11-13
        # 3. Strength of the matches (how close to exact harmonics)
        mean_error = np.mean([abs(s - h) for s, h in zip(matched_scales, harmonics[:len(matched_scales)])])
        score = matches * (1 + 1/(mean_error + 1e-6))  # Add small epsilon to avoid divide by zero
        
        # Extra weight for scales in the expected range
        if 10 <= fundamental <= 14:
            score *= 1.2
            
        if score > best_score:
            best_score = score
            best_fundamental = fundamental
            best_harmonics = matched_scales
    
    if best_fundamental is None:
        return None
        
    return best_fundamental

def estimate_scale_fft(image, asset_idx=None, debug_dir=None):
    """
    Estimate the pixel art scale factor using 2D FFT analysis.
    
    Args:
        image: PIL Image in RGBA format
        asset_idx: Optional index for debug output naming
        debug_dir: Optional directory for debug output
        
    Returns:
        int: Estimated scale factor, or None if estimation fails
    """
    # Convert to grayscale numpy array
    image_np = np.array(image.convert('L'))
    
    # Apply light Gaussian blur to reduce high-frequency noise
    image_np = gaussian_filter(image_np, sigma=0.5)
    
    # Apply FFT
    fft = np.fft.fft2(image_np)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    # Find the center
    center_y, center_x = np.array(magnitude.shape) // 2
    
    # Create mask to exclude central region (DC component)
    mask = np.ones_like(magnitude)
    exclude_radius = min(center_y, center_x) // 16  # Smaller exclusion zone
    y, x = np.ogrid[-center_y:magnitude.shape[0]-center_y, -center_x:magnitude.shape[1]-center_x]
    mask_area = x*x + y*y <= exclude_radius*exclude_radius
    mask[mask_area] = 0
    
    # Also mask high frequencies (outer region)
    outer_radius = min(center_y, center_x) // 2  # Only look at lower frequencies
    outer_mask = x*x + y*y >= outer_radius*outer_radius
    mask[outer_mask] = 0
    
    # Apply mask to magnitude spectrum
    masked_magnitude = magnitude * mask
    
    # Save FFT magnitude spectrum for debugging
    if debug_dir is not None:
        # Use log scale for visualization only
        vis_mag = np.log(1 + masked_magnitude)
        vis_mag = cv2.normalize(vis_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        prefix = f"asset_{asset_idx}_" if asset_idx is not None else ""
        debug_path = os.path.join(debug_dir, f"{prefix}fft_magnitude.png")
        Image.fromarray(vis_mag).save(debug_path)
        
        # Create colored version for peak visualization
        vis_mag_color = cv2.cvtColor(vis_mag, cv2.COLOR_GRAY2RGB)
    
    # Find peaks in 2D magnitude spectrum using linear magnitude
    # This gives more weight to stronger frequency components
    peaks = peak_local_max(
        masked_magnitude,
        min_distance=3,        # Allow closer peaks
        threshold_rel=0.2,     # Lower threshold to find more peaks
        num_peaks=32,         # Look for more peaks
        exclude_border=False
    )
    
    if len(peaks) < 4:
        print("Warning: Not enough peaks found in FFT spectrum")
        return None
    
    # Calculate distances and collect peak information
    peak_info = []
    for peak_y, peak_x in peaks:
        dist = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
        if dist > exclude_radius:  # Skip peaks too close to center
            # Scale calculation based on spatial period:
            # The distance in frequency space represents half a period
            # (from -?? to ?? in normalized frequency)
            # So we multiply by 2 to get the full period
            scale_h = image_np.shape[0] / (dist * 2)  # Period = N/freq
            scale_w = image_np.shape[1] / (dist * 2)
            scale_est = (scale_h + scale_w) / 2
            
            # Record peak strength for potential weighting
            strength = masked_magnitude[peak_y, peak_x]
            peak_info.append((dist, scale_est, peak_y, peak_x, strength))
    
    if not peak_info:
        print("Warning: No valid scale estimates from FFT peaks")
        return None
    
    # Sort by peak strength and get scale estimates
    peak_info.sort(key=lambda x: x[4], reverse=True)  # Sort by strength
    scale_estimates = [p[1] for p in peak_info]
    
    # Find fundamental scale considering harmonics
    fundamental_scale = find_harmonic_groups(scale_estimates)
    if fundamental_scale is None:
        return None
    
    # Round to nearest integer
    scale = int(round(fundamental_scale))
    
    if debug_dir is not None:
        # Draw all peaks in red
        for _, _, py, px, _ in peak_info:
            cv2.circle(vis_mag_color, (px, py), 2, (0, 0, 255), -1)
        
        # Draw harmonic relationships
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green, Yellow, Cyan
        drawn_peaks = set()
        
        for harmonic in [1, 2, 3]:
            expected_scale = fundamental_scale * harmonic
            for dist, scale_est, py, px, _ in peak_info:
                if abs(scale_est - expected_scale) / expected_scale < 0.12:
                    if (py, px) not in drawn_peaks:
                        color = colors[min(harmonic-1, len(colors)-1)]
                        cv2.circle(vis_mag_color, (px, py), 4, color, -1)
                        cv2.line(vis_mag_color, (center_x, center_y), (px, py), color, 1)
                        drawn_peaks.add((py, px))
        
        # Save magnitude spectrum with marked peaks
        debug_path = os.path.join(debug_dir, f"{prefix}fft_magnitude_peaks.png")
        Image.fromarray(vis_mag_color).save(debug_path)
    
    # Print scale estimates for debugging
    print(f"FFT Analysis found {len(peak_info)} peaks")
    print("Raw scale estimates (from strongest peaks):", 
          ", ".join(f"{s:.1f}" for s in scale_estimates[:8]) +
          ("..." if len(scale_estimates) > 8 else ""))
    print(f"Detected fundamental scale: {fundamental_scale:.1f}")
    print(f"Final rounded scale: {scale}")
    
    return scale
