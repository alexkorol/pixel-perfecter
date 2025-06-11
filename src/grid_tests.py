"""
Grid Detection and Validation Tests

This module implements the testing methodology documented in notes/testing_methodology.md
for validating grid detection accuracy and artifact removal in pixel art reconstruction.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import find_peaks
from collections import Counter
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class GridDetectionTests:
    """Test suite for grid size and offset accuracy."""
    
    def __init__(self, image: np.ndarray):
        """
        Initialize with input image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
    def visual_grid_overlay(self, cell_size: int, offset: Tuple[int, int] = (0, 0), 
                           color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
        """
        Draw detected grid lines on top of the original image.
        
        This is the most important sanity check - allows instant visual verification
        of whether the grid aligns with the visual blocks.
        
        Args:
            cell_size: Size of each grid cell in pixels
            offset: (x_offset, y_offset) grid alignment offset
            color: RGB color for grid lines
            
        Returns:
            Image with grid overlay
        """
        overlay = self.image.copy()
        height, width = overlay.shape[:2]
        x_offset, y_offset = offset
        
        # Draw vertical lines
        for x in range(x_offset, width, cell_size):
            cv2.line(overlay, (x, 0), (x, height), color, 1)
            
        # Draw horizontal lines  
        for y in range(y_offset, height, cell_size):
            cv2.line(overlay, (0, y), (width, y), color, 1)
            
        return overlay
    
    def boundary_edge_scoring(self, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> float:
        """
        Count edge transitions that align with grid boundaries.
        
        Higher scores indicate better grid alignment with actual image structure.
        
        Args:
            cell_size: Size of each grid cell
            offset: Grid offset
            
        Returns:
            Score representing grid boundary alignment quality
        """
        height, width = self.gray.shape
        x_offset, y_offset = offset
        score = 0
        
        # Check vertical grid lines
        for x in range(x_offset, width - 1, cell_size):
            if x < width - 1:
                # Count color differences across this vertical line
                left_col = self.gray[:, x]
                right_col = self.gray[:, x + 1]
                score += np.sum(np.abs(left_col.astype(int) - right_col.astype(int)) > 10)
        
        # Check horizontal grid lines
        for y in range(y_offset, height - 1, cell_size):
            if y < height - 1:
                # Count color differences across this horizontal line
                top_row = self.gray[y, :]
                bottom_row = self.gray[y + 1, :]
                score += np.sum(np.abs(top_row.astype(int) - bottom_row.astype(int)) > 10)
                
        return score
    
    def edge_projection_peak_analysis(self, edge_threshold: int = 50) -> Tuple[int, int]:
        """
        Find cell size using edge projection and peak spacing analysis.
        
        This is the most robust method for finding true cell_size, even with
        noise, anti-aliasing, and non-integer scaling.
        
        Args:
            edge_threshold: Threshold for edge detection
            
        Returns:
            (horizontal_cell_size, vertical_cell_size) tuple
        """
        # Edge detection
        edges = cv2.Canny(self.gray, edge_threshold, edge_threshold * 2)
        
        # Create projections by summing edges along each axis
        horizontal_projection = np.sum(edges, axis=0)  # Sum along rows -> column peaks
        vertical_projection = np.sum(edges, axis=1)    # Sum along columns -> row peaks
        
        def find_modal_spacing(projection: np.ndarray) -> int:
            """Find the most common spacing between peaks."""
            # Find peaks in projection
            peaks, _ = find_peaks(projection, height=np.max(projection) * 0.1)
            
            if len(peaks) < 2:
                return 0
                
            # Calculate spacings between consecutive peaks
            spacings = np.diff(peaks)
            
            if len(spacings) == 0:
                return 0
                
            # Find most common spacing
            spacing_counts = Counter(spacings)
            modal_spacing = spacing_counts.most_common(1)[0][0]
            
            return modal_spacing
        
        h_cell_size = find_modal_spacing(horizontal_projection)
        v_cell_size = find_modal_spacing(vertical_projection)
        
        return h_cell_size, v_cell_size
    
    def intra_block_variance(self, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> float:
        """
        Calculate color variance within grid blocks.
        
        Correct grid size should have lowest internal variance as it aligns
        with flat-color areas. Perfect for distinguishing correct grid from sub-multiples.
        
        Args:
            cell_size: Size of each grid cell
            offset: Grid offset
            
        Returns:
            Average variance within all blocks
        """
        height, width = self.gray.shape
        x_offset, y_offset = offset
        variances = []
        
        # Analyze each grid block
        for y in range(y_offset, height - cell_size, cell_size):
            for x in range(x_offset, width - cell_size, cell_size):
                # Extract block
                block = self.gray[y:y+cell_size, x:x+cell_size]
                
                # Calculate variance within this block
                if block.size > 0:
                    variances.append(np.var(block))
        
        return np.mean(variances) if variances else float('inf')


class ArtifactTests:
    """Test suite for artifacts and stray pixel detection/correction."""
    
    def __init__(self, original: np.ndarray, processed: np.ndarray):
        """
        Initialize with original and processed images.
        
        Args:
            original: Original input image
            processed: Processed/snapped output image
        """
        self.original = original
        self.processed = processed
        
    def logical_neighbor_majority_vote(self, cell_size: int, 
                                     threshold: float = 0.7) -> np.ndarray:
        """
        Fix isolated single-block errors using neighbor majority voting.
        
        Operates on logical (block-by-block) level to smooth out stray pixels
        without affecting overall structure.
        
        Args:
            cell_size: Size of logical blocks
            threshold: Minimum fraction of neighbors needed for replacement
            
        Returns:
            Corrected image
        """
        if len(self.processed.shape) == 3:
            height, width, channels = self.processed.shape
        else:
            height, width = self.processed.shape
            channels = 1
            
        corrected = self.processed.copy()
        
        # Work on logical grid
        for y in range(0, height - cell_size + 1, cell_size):
            for x in range(0, width - cell_size + 1, cell_size):
                # Get current block color (mode of block)
                if channels == 3:
                    block = self.processed[y:y+cell_size, x:x+cell_size]
                    # Convert to single values for comparison
                    block_colors = block.reshape(-1, 3)
                    current_color = tuple(np.median(block_colors, axis=0).astype(int))
                else:
                    block = self.processed[y:y+cell_size, x:x+cell_size]
                    current_color = int(np.median(block))
                
                # Get neighbor block colors (4-connected)
                neighbor_colors = []
                for dy, dx in [(-cell_size, 0), (cell_size, 0), (0, -cell_size), (0, cell_size)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height - cell_size + 1 and 0 <= nx < width - cell_size + 1:
                        if channels == 3:
                            neighbor_block = self.processed[ny:ny+cell_size, nx:nx+cell_size]
                            neighbor_colors_flat = neighbor_block.reshape(-1, 3)
                            neighbor_color = tuple(np.median(neighbor_colors_flat, axis=0).astype(int))
                        else:
                            neighbor_block = self.processed[ny:ny+cell_size, nx:nx+cell_size]
                            neighbor_color = int(np.median(neighbor_block))
                        neighbor_colors.append(neighbor_color)
                
                # Find majority neighbor color
                if neighbor_colors:
                    color_counts = Counter(neighbor_colors)
                    majority_color, count = color_counts.most_common(1)[0]
                    
                    # Replace if current color is outlier
                    if count / len(neighbor_colors) >= threshold and majority_color != current_color:
                        if channels == 3:
                            corrected[y:y+cell_size, x:x+cell_size] = majority_color
                        else:
                            corrected[y:y+cell_size, x:x+cell_size] = majority_color
        
        return corrected
    
    def high_resolution_difference_overlay(self, cell_size: int) -> np.ndarray:
        """
        Create high-contrast difference overlay for objective accuracy validation.
        
        This is the ultimate objective test - visually flags any deviation and helps
        identify exactly where the algorithm is failing.
        
        Args:
            cell_size: Cell size for upscaling processed image
            
        Returns:
            Difference overlay image (magenta on black)
        """
        # Upscale processed image back to original resolution
        if len(self.processed.shape) == 3:
            processed_height, processed_width = self.processed.shape[:2]
        else:
            processed_height, processed_width = self.processed.shape
            
        original_height, original_width = self.original.shape[:2]
        
        # Use nearest neighbor upscaling
        upscaled = cv2.resize(self.processed, 
                             (original_width, original_height), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Compute per-pixel difference
        if len(self.original.shape) == 3 and len(upscaled.shape) == 3:
            diff = np.abs(self.original.astype(float) - upscaled.astype(float))
            diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))
        else:
            # Convert to grayscale if needed
            orig_gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY) if len(self.original.shape) == 3 else self.original
            up_gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY) if len(upscaled.shape) == 3 else upscaled
            diff_magnitude = np.abs(orig_gray.astype(float) - up_gray.astype(float))
        
        # Create high-contrast overlay (magenta on black)
        overlay = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        
        # Threshold for significant differences
        threshold = 30
        mask = diff_magnitude > threshold
        
        # Set differences to magenta
        overlay[mask] = [255, 0, 255]  # Magenta
        
        return overlay
    
    def block_difference_stray_filter(self, cell_size: int, 
                                    difference_threshold: float = 50.0) -> np.ndarray:
        """
        Advanced stray detection that distinguishes expected vs unexpected differences.
        
        Flags blocks as strays only if average difference exceeds threshold,
        distinguishing between expected differences (anti-aliasing removal) and
        unexpected differences (truly out-of-place blocks).
        
        Args:
            cell_size: Size of blocks to analyze
            difference_threshold: Threshold for flagging blocks as strays
            
        Returns:
            Binary mask where 1 indicates stray blocks
        """
        height, width = self.original.shape[:2]
        
        # Upscale processed to match original resolution
        upscaled = cv2.resize(self.processed, (width, height), 
                             interpolation=cv2.INTER_NEAREST)
        
        stray_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Analyze each block
        for y in range(0, height - cell_size + 1, cell_size):
            for x in range(0, width - cell_size + 1, cell_size):
                # Extract corresponding blocks
                orig_block = self.original[y:y+cell_size, x:x+cell_size]
                proc_block = upscaled[y:y+cell_size, x:x+cell_size]
                
                # Calculate average difference in this block
                if len(orig_block.shape) == 3:
                    block_diff = np.mean(np.abs(orig_block.astype(float) - proc_block.astype(float)))
                else:
                    block_diff = np.mean(np.abs(orig_block.astype(float) - proc_block.astype(float)))
                
                # Flag as stray if difference exceeds threshold
                if block_diff > difference_threshold:
                    stray_mask[y:y+cell_size, x:x+cell_size] = 1
        
        return stray_mask


def run_comprehensive_grid_analysis(image_path: str, 
                                  candidate_cell_sizes: List[int] = None,
                                  save_debug: bool = True) -> dict:
    """
    Run comprehensive grid analysis using all test methods.
    
    Args:
        image_path: Path to input image
        candidate_cell_sizes: List of cell sizes to test
        save_debug: Whether to save debug visualizations
        
    Returns:
        Dictionary with analysis results
    """
    if candidate_cell_sizes is None:
        candidate_cell_sizes = [8, 16, 24, 32, 48, 64]
    
    # Load image
    image = np.array(Image.open(image_path))
    tests = GridDetectionTests(image)
    
    results = {
        'image_path': image_path,
        'image_shape': image.shape,
        'candidate_sizes': candidate_cell_sizes,
        'analysis': {}
    }
    
    # 1. Edge projection peak analysis (most robust)
    h_cell, v_cell = tests.edge_projection_peak_analysis()
    results['edge_projection_result'] = {'horizontal': h_cell, 'vertical': v_cell}
    
    # 2. Test each candidate cell size
    for cell_size in candidate_cell_sizes:
        analysis = {}
        
        # Boundary edge scoring
        edge_score = tests.boundary_edge_scoring(cell_size)
        analysis['edge_score'] = edge_score
        
        # Intra-block variance
        variance = tests.intra_block_variance(cell_size)
        analysis['intra_block_variance'] = variance
        
        # Visual grid overlay
        if save_debug:
            overlay = tests.visual_grid_overlay(cell_size)
            overlay_path = f"debug_grid_overlay_{cell_size}px.png"
            Image.fromarray(overlay).save(overlay_path)
            analysis['overlay_path'] = overlay_path
        
        results['analysis'][cell_size] = analysis
    
    # 3. Recommend best cell size
    # Prioritize edge projection result, fall back to lowest variance
    if h_cell > 0 and v_cell > 0:
        if h_cell == v_cell:
            results['recommended_cell_size'] = h_cell
        else:
            results['recommended_cell_size'] = min(h_cell, v_cell)  # Conservative choice
    else:
        # Fall back to variance analysis
        variances = {size: results['analysis'][size]['intra_block_variance'] 
                    for size in candidate_cell_sizes}
        best_size = min(variances.keys(), key=lambda k: variances[k])
        results['recommended_cell_size'] = best_size
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Grid Detection and Validation Tests")
    print("===================================")
    
    # This would be used with an actual image file
    # results = run_comprehensive_grid_analysis("input/test_image.png")
    # print(f"Recommended cell size: {results['recommended_cell_size']}")
    # print(f"Edge projection detected: {results['edge_projection_result']}")
    
    print("Import this module and use with your images:")
    print("from grid_tests import GridDetectionTests, ArtifactTests, run_comprehensive_grid_analysis")
