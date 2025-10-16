#!/usr/bin/env python3
"""
Pixel Art Reconstruction Module

Takes an AI-generated pseudo-pixel art image and reconstructs it into 
true 1:1 pixel art using a unified three-step pipeline.
"""

import numpy as np
from PIL import Image
import cv2
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter
from typing import Tuple, Optional


class PixelArtReconstructor:
    """
    Unified pixel art reconstruction pipeline.
    
    Implements a robust three-step process:
    1. Detect grid parameters using edge projection and intra-block variance
    2. Snap to grid using modal color per cell
    3. Refine artifacts with neighbor majority vote
    """
    
    def __init__(self, image: np.ndarray):
        """
        Initialize with input image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        self.original = image
        self.height, self.width = image.shape[:2]
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Pipeline results
        self.cell_size = None
        self.offset = None
        self.pixel_art = None
    
    def run(self) -> np.ndarray:
        """
        Execute the complete reconstruction pipeline.
        
        Returns:
            Reconstructed pixel art as numpy array
        """
        # Step 1: Detect grid parameters
        self.cell_size, self.offset = self._detect_grid_parameters()
        
        # Step 2: Snap to grid using modal colors
        self.pixel_art = self._snap_to_grid()
        
        # Step 3: Refine artifacts using neighbor majority vote
        self.pixel_art = self._refine_artifacts()
        
        return self.pixel_art
    
    def _detect_grid_parameters(self) -> Tuple[int, Tuple[int, int]]:
        """
        Step 1: Detect grid cell size and offset using edge projection and variance analysis.
        
        Returns:
            (cell_size, (x_offset, y_offset))
        """
        # First try edge projection peak analysis (most robust)
        cell_size_from_edges = self._edge_projection_analysis()
        
        if cell_size_from_edges > 0:
            # Use edge projection result and find best offset
            best_offset = self._find_best_offset(cell_size_from_edges)
            return cell_size_from_edges, best_offset
        
        # Fallback: test candidate sizes with variance analysis
        candidates = [8, 16, 24, 32, 48, 64]
        best_cell_size = None
        best_variance = float('inf')
        best_offset = (0, 0)
        
        for cell_size in candidates:
            # Find best offset for this cell size
            offset = self._find_best_offset(cell_size)
            variance = self._intra_block_variance(cell_size, offset)
            
            if variance < best_variance:
                best_variance = variance
                best_cell_size = cell_size
                best_offset = offset
        
        return best_cell_size, best_offset
    
    def _edge_projection_analysis(self) -> int:
        """
        Find cell size using edge projection and peak spacing analysis.
        
        Returns:
            Cell size, or 0 if detection failed
        """
        # Edge detection
        edges = cv2.Canny(self.gray, 50, 100)
        
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
        
        # Return consistent size, or 0 if inconsistent
        if h_cell_size > 0 and v_cell_size > 0:
            if h_cell_size == v_cell_size:
                return h_cell_size
            else:
                # Use the smaller one as conservative choice
                return min(h_cell_size, v_cell_size)
        elif h_cell_size > 0:
            return h_cell_size
        elif v_cell_size > 0:
            return v_cell_size
        else:
            return 0
    
    def _find_best_offset(self, cell_size: int) -> Tuple[int, int]:
        """
        Find optimal grid offset using boundary edge scoring.
        
        Args:
            cell_size: Grid cell size
            
        Returns:
            (x_offset, y_offset) with best boundary alignment
        """
        best_score = -1
        best_offset = (0, 0)
        
        # Search within one cell size
        for x_offset in range(0, cell_size):
            for y_offset in range(0, cell_size):
                score = self._boundary_edge_score(cell_size, (x_offset, y_offset))
                
                if score > best_score:
                    best_score = score
                    best_offset = (x_offset, y_offset)
        
        return best_offset
    
    def _boundary_edge_score(self, cell_size: int, offset: Tuple[int, int]) -> float:
        """
        Score how well grid boundaries align with image edges.
        
        Args:
            cell_size: Size of each grid cell
            offset: Grid offset
            
        Returns:
            Score representing boundary alignment quality
        """
        x_offset, y_offset = offset
        score = 0
        
        # Check vertical grid lines
        for x in range(x_offset, self.width - 1, cell_size):
            if x < self.width - 1:
                # Count color differences across this vertical line
                left_col = self.gray[:, x]
                right_col = self.gray[:, x + 1]
                score += np.sum(np.abs(left_col.astype(int) - right_col.astype(int)) > 10)
        
        # Check horizontal grid lines
        for y in range(y_offset, self.height - 1, cell_size):
            if y < self.height - 1:
                # Count color differences across this horizontal line
                top_row = self.gray[y, :]
                bottom_row = self.gray[y + 1, :]
                score += np.sum(np.abs(top_row.astype(int) - bottom_row.astype(int)) > 10)
                
        return score
    
    def _intra_block_variance(self, cell_size: int, offset: Tuple[int, int]) -> float:
        """
        Calculate average color variance within grid blocks.
        
        Args:
            cell_size: Size of each grid cell
            offset: Grid offset
            
        Returns:
            Average variance within all blocks
        """
        x_offset, y_offset = offset
        variances = []
        
        # Analyze each grid block
        for y in range(y_offset, self.height - cell_size, cell_size):
            for x in range(x_offset, self.width - cell_size, cell_size):
                # Extract block
                block = self.gray[y:y+cell_size, x:x+cell_size]
                
                # Calculate variance within this block
                if block.size > 0:
                    variances.append(np.var(block))
        
        return np.mean(variances) if variances else float('inf')
    
    def _snap_to_grid(self) -> np.ndarray:
        """
        Step 2: Snap image to grid using modal color per cell.
        
        Returns:
            Grid-snapped pixel art
        """
        x_offset, y_offset = self.offset
        
        # Calculate logical grid dimensions
        logical_width = (self.width - x_offset + self.cell_size - 1) // self.cell_size
        logical_height = (self.height - y_offset + self.cell_size - 1) // self.cell_size
        
        # Create pixel art array
        if len(self.original.shape) == 3:
            pixel_art = np.zeros((logical_height, logical_width, 3), dtype=np.uint8)
        else:
            pixel_art = np.zeros((logical_height, logical_width), dtype=np.uint8)
        
        # Extract modal color for each grid cell
        for row in range(logical_height):
            for col in range(logical_width):
                # Calculate pixel coordinates for this block
                y_start = y_offset + row * self.cell_size
                y_end = min(y_start + self.cell_size, self.height)
                x_start = x_offset + col * self.cell_size  
                x_end = min(x_start + self.cell_size, self.width)
                
                # Skip if block is completely outside image
                if y_start >= self.height or x_start >= self.width:
                    continue
                
                # Extract block (may be partial at edges)
                block = self.original[y_start:y_end, x_start:x_end]
                
                # Skip empty blocks
                if block.size == 0:
                    continue
                
                # Find modal color
                if len(self.original.shape) == 3:
                    # For RGB, find mode of each channel
                    modal_color = []
                    for channel in range(3):
                        channel_data = block[:, :, channel].flatten()
                        mode_result = stats.mode(channel_data, keepdims=True)
                        modal_color.append(mode_result.mode[0])
                    pixel_art[row, col] = modal_color
                else:
                    # For grayscale
                    block_data = block.flatten()
                    mode_result = stats.mode(block_data, keepdims=True)
                    pixel_art[row, col] = mode_result.mode[0]
        
        return pixel_art
    
    def _refine_artifacts(self) -> np.ndarray:
        """
        Step 3: Refine artifacts using neighbor majority vote.
        
        Returns:
            Refined pixel art with artifacts corrected
        """
        if self.pixel_art is None:
            raise ValueError("Must run snap_to_grid first")
        
        refined = self.pixel_art.copy()
        height, width = refined.shape[:2]
        
        # Apply neighbor majority vote filter
        for row in range(height):
            for col in range(width):
                # Get current color
                if len(refined.shape) == 3:
                    current_color = tuple(refined[row, col])
                else:
                    current_color = refined[row, col]
                
                # Get neighbor colors (4-connected)
                neighbor_colors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = row + dy, col + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if len(refined.shape) == 3:
                            neighbor_color = tuple(refined[ny, nx])
                        else:
                            neighbor_color = refined[ny, nx]
                        neighbor_colors.append(neighbor_color)
                
                # Find majority neighbor color
                if neighbor_colors:
                    color_counts = Counter(neighbor_colors)
                    majority_color, count = color_counts.most_common(1)[0]
                    
                    # Replace if current color is outlier (threshold 0.7 = 3/4 neighbors)
                    if count / len(neighbor_colors) >= 0.7 and majority_color != current_color:
                        if len(refined.shape) == 3:
                            refined[row, col] = majority_color
                        else:
                            refined[row, col] = majority_color

        return refined


def validate_reconstruction(image_path: str) -> None:
    """
    Validate the pixel art reconstruction pipeline.
    
    Args:
        image_path: Path to input image
    """
    import os
    from grid_tests import GridVisualizer, ReconstructionAnalyzer
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"Validating reconstruction pipeline for: {image_path}")
    print("=" * 60)
    
    # Load original image
    try:
        original = np.array(Image.open(image_path))
        print(f"Loaded image: {original.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Run unified reconstruction pipeline
    print("Running reconstruction pipeline...")
    try:
        reconstructor = PixelArtReconstructor(original)
        pixel_art = reconstructor.run()
        
        print(f"Detected cell size: {reconstructor.cell_size}px")
        print(f"Detected offset: {reconstructor.offset}")
        print(f"Reconstructed size: {pixel_art.shape}")
        
        # Calculate reduction factor
        orig_pixels = original.shape[0] * original.shape[1]
        recon_pixels = pixel_art.shape[0] * pixel_art.shape[1]
        reduction_factor = orig_pixels / recon_pixels
        print(f"Pixel reduction: {reduction_factor:.1f}x")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "output/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create visual grid overlay on original
    print("Creating grid overlay...")
    visualizer = GridVisualizer(original)
    grid_overlay = visualizer.create_grid_overlay(
        reconstructor.cell_size, 
        reconstructor.offset,
        color=(255, 0, 255)  # Magenta grid lines
    )
    
    grid_overlay_path = os.path.join(output_dir, f"{base_name}_grid_overlay.png")
    Image.fromarray(grid_overlay).save(grid_overlay_path)
    print(f"Saved grid overlay: {grid_overlay_path}")
    
    # 2. Create high-res difference overlay between original and upscaled output
    print("Creating difference overlay...")
    analyzer = ReconstructionAnalyzer(original, pixel_art)
    diff_overlay = analyzer.create_difference_overlay(reconstructor.cell_size)
    
    diff_overlay_path = os.path.join(output_dir, f"{base_name}_difference_overlay.png")
    Image.fromarray(diff_overlay).save(diff_overlay_path)
    print(f"Saved difference overlay: {diff_overlay_path}")
    
    # Also save the pure pixel art for reference
    pixel_art_path = os.path.join(output_dir, f"{base_name}_pixel_art.png")
    Image.fromarray(pixel_art).save(pixel_art_path)
    print(f"Saved pixel art: {pixel_art_path}")
    
    print("\nVALIDATION COMPLETE")
    print("=" * 60)
    print("Check the following outputs:")
    print(f"1. Grid overlay (shows detected grid alignment): {grid_overlay_path}")
    print(f"2. Difference overlay (shows reconstruction errors): {diff_overlay_path}")
    print(f"3. Pure pixel art result: {pixel_art_path}")
    print("\nGrid overlay should align with visual blocks.")
    print("Difference overlay should be mostly black (minimal magenta = good reconstruction).")


def process_all_images() -> None:
    """
    Process all images in the input directory.
    """
    import os
    import glob
    
    input_dir = "input"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        return
    
    # Find all image files in input directory
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.gif"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in '{input_dir}' directory!")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF")
        return
    
    print(f"Found {len(image_files)} image(s) to process:")
    for img in image_files:
        print(f"  - {os.path.basename(img)}")
    print()
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        print("-" * 80)
        validate_reconstruction(image_path)
    
    print(f"\n???? BATCH PROCESSING COMPLETE")
    print(f"Processed {len(image_files)} images successfully!")
    print("Check the 'output/validation' directory for results.")


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        # Single image mode (backward compatibility)
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            validate_reconstruction(image_path)
        else:
            print(f"Error: Image file '{image_path}' not found!")
    else:
        # Batch mode - process all images in input directory
        print("Pixel Art Reconstructor - Unified Pipeline")
        print("Processing all images in 'input/' directory...\n")
        process_all_images()
