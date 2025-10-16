#!/usr/bin/env python3
"""
Pixel Art Reconstruction Module

Takes an AI-generated pseudo-pixel art image and reconstructs it into 
true 1:1 pixel art using the detected grid parameters.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from PIL import Image
from scipy import stats
from grid_tests import GridDetectionTests


class PixelArtReconstructor:
    """Reconstructs true pixel art from AI-generated pseudo-pixel art."""
    
    def __init__(self, image: np.ndarray):
        """
        Initialize with input image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        self.original = image
        self.height, self.width = image.shape[:2]
        
    def extract_grid_blocks(self, cell_size: int, offset: tuple = (0, 0)) -> tuple:
        """
        Extract color information from each grid block.
        
        Args:
            cell_size: Size of each grid cell
            offset: (x_offset, y_offset) grid alignment offset
            
        Returns:
            (grid_colors, logical_width, logical_height) where grid_colors is 
            a 2D array of the modal color for each block
        """
        x_offset, y_offset = offset
        
        # Calculate logical grid dimensions
        logical_width = (self.width - x_offset) // cell_size
        logical_height = (self.height - y_offset) // cell_size
        
        # Extract modal color for each block
        if len(self.original.shape) == 3:
            grid_colors = np.zeros((logical_height, logical_width, 3), dtype=np.uint8)
        else:
            grid_colors = np.zeros((logical_height, logical_width), dtype=np.uint8)
        
        for row in range(logical_height):
            for col in range(logical_width):
                # Calculate pixel coordinates for this block
                y_start = y_offset + row * cell_size
                y_end = y_start + cell_size
                x_start = x_offset + col * cell_size  
                x_end = x_start + cell_size
                
                # Extract block
                block = self.original[y_start:y_end, x_start:x_end]
                
                # Find modal color
                if len(self.original.shape) == 3:
                    # For RGB, find mode of each channel
                    modal_color = []
                    for channel in range(3):
                        channel_data = block[:, :, channel].flatten()
                        mode_result = stats.mode(channel_data, keepdims=True)
                        modal_color.append(mode_result.mode[0])
                    grid_colors[row, col] = modal_color
                else:
                    # For grayscale
                    block_data = block.flatten()
                    mode_result = stats.mode(block_data, keepdims=True)
                    grid_colors[row, col] = mode_result.mode[0]
        
        return grid_colors, logical_width, logical_height
    
    def find_optimal_offset(self, cell_size: int, search_range: int = None) -> tuple:
        """
        Find the optimal grid offset using boundary edge scoring.
        
        Args:
            cell_size: Grid cell size
            search_range: Range to search for offsets (default: cell_size)
            
        Returns:
            (best_x_offset, best_y_offset) tuple
        """
        if search_range is None:
            search_range = cell_size
            
        tests = GridDetectionTests(self.original)
        
        best_score = -1
        best_offset = (0, 0)
        
        # Try different offsets
        for x_offset in range(0, min(search_range, cell_size)):
            for y_offset in range(0, min(search_range, cell_size)):
                score = tests.boundary_edge_scoring(cell_size, (x_offset, y_offset))
                
                if score > best_score:
                    best_score = score
                    best_offset = (x_offset, y_offset)
        
        return best_offset
    
    def reconstruct_pixel_art(self, cell_size: int, offset: tuple = None) -> np.ndarray:
        """
        Reconstruct true pixel art from the original image.
        
        Args:
            cell_size: Size of each grid cell in the original
            offset: Grid offset, or None to auto-detect
            
        Returns:
            Reconstructed pixel art as numpy array
        """
        # Find optimal offset if not provided
        if offset is None:
            offset = self.find_optimal_offset(cell_size)
            print(f"Auto-detected offset: {offset}")
        
        # Extract grid blocks
        grid_colors, logical_width, logical_height = self.extract_grid_blocks(cell_size, offset)
        
        print(f"Logical dimensions: {logical_width}x{logical_height}")
        print(f"Original dimensions: {self.width}x{self.height}")
        print(f"Reduction factor: {self.width//logical_width:.1f}x")
        
        return grid_colors
    
    def upscale_pixel_art(self, pixel_art: np.ndarray, scale_factor: int) -> np.ndarray:
        """
        Upscale pixel art using nearest neighbor (crisp pixels).
        
        Args:
            pixel_art: The reconstructed pixel art
            scale_factor: How many times to scale up each pixel
            
        Returns:
            Upscaled image
        """
        if len(pixel_art.shape) == 3:
            h, w, c = pixel_art.shape
            upscaled = np.zeros((h * scale_factor, w * scale_factor, c), dtype=np.uint8)
        else:
            h, w = pixel_art.shape
            upscaled = np.zeros((h * scale_factor, w * scale_factor), dtype=np.uint8)
        
        # Nearest neighbor upscaling
        for y in range(h):
            for x in range(w):
                y_start = y * scale_factor
                y_end = y_start + scale_factor
                x_start = x * scale_factor
                x_end = x_start + scale_factor
                
                upscaled[y_start:y_end, x_start:x_end] = pixel_art[y, x]
        
        return upscaled


def process_image(image_path: str, cell_size: int, output_path: str = None):
    """
    Process a single image to extract pixel art.
    
    Args:
        image_path: Path to input image
        cell_size: Grid cell size to use
        output_path: Where to save result (optional)
    """
    print(f"Processing: {image_path}")
    print(f"Using cell size: {cell_size}px")
    
    # Load image
    original = np.array(Image.open(image_path))
    
    # Reconstruct
    reconstructor = PixelArtReconstructor(original)
    pixel_art = reconstructor.reconstruct_pixel_art(cell_size)
    
    # Save the pure pixel art (small version)
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{base_name}_reconstructed_{cell_size}px.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(pixel_art).save(output_path)
    print(f"Saved pure pixel art: {output_path}")
    
    # Also save an upscaled version for easier viewing
    upscaled_path = output_path.replace('.png', '_upscaled.png')
    
    # Calculate good upscale factor
    original_size = max(original.shape[:2])
    pixel_art_size = max(pixel_art.shape[:2])
    scale_factor = max(1, original_size // (pixel_art_size * 4))  # Target ~1/4 original size
    
    upscaled = reconstructor.upscale_pixel_art(pixel_art, scale_factor)
    Image.fromarray(upscaled).save(upscaled_path)
    print(f"Saved upscaled version: {upscaled_path}")
    
    return pixel_art, upscaled


def main():
    """Process all test images with their optimal cell sizes."""
    
    print("PIXEL ART RECONSTRUCTION")
    print("========================")
    
    # Based on our analysis, use these cell sizes:
    test_cases = [
        ("input/20250610_1843_Red Triangle Pixel Art_simple_compose_01jxeafmtye8v8zpqyn290t9nb.png", 32),  # Try 32 first, 8 might be too small
        ("input/20250610_2049_Green Smiley Pixel Art_simple_compose_01jxehphg7fjhrc4gy66858g3k.png", 32),  # Edge projection suggested this
        ("input/20250610_2037_Pixel Gold Star_simple_compose_01jxegz6xsfk6bc9dng8a92zhs.png", 16),       # 8 might be too small
        ("input/20250329_2258_Flytrap Pollen Close-Up_remix_01jqjt2qw9fmvaj968ftkvmvfx.png", 28),        # Edge projection was 28x28
    ]
    
    results = []
    
    for image_path, cell_size in test_cases:
        if os.path.exists(image_path):
            try:
                pixel_art, upscaled = process_image(image_path, cell_size)
                results.append((image_path, cell_size, pixel_art.shape))
                print(f"??? Success\\n")
            except Exception as e:
                print(f"??? Error: {e}\\n")
        else:
            print(f"??? File not found: {image_path}\\n")
    
    print("RECONSTRUCTION COMPLETE")
    print("=======================")
    
    for image_path, cell_size, final_shape in results:
        name = os.path.basename(image_path).split('_')[2]
        print(f"{name}: {cell_size}px cell -> {final_shape[1]}x{final_shape[0]} pixel art")
    
    print("\\nCheck output/ directory for:")
    print("- *_reconstructed_XXpx.png (pure pixel art)")
    print("- *_reconstructed_XXpx_upscaled.png (easier to view)")


if __name__ == "__main__":
    main()
