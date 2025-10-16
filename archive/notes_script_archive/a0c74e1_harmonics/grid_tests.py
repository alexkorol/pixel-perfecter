"""
Grid Visualization and Analysis Tools

This module provides visualization tools for validating pixel art reconstruction results.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple


class GridVisualizer:
    """Tools for visualizing grid alignment and reconstruction accuracy."""
    
    def __init__(self, image: np.ndarray):
        """
        Initialize with input image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        self.image = image
        
    def create_grid_overlay(self, cell_size: int, offset: Tuple[int, int] = (0, 0), 
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


class ReconstructionAnalyzer:
    """Tools for analyzing reconstruction accuracy."""
    
    def __init__(self, original: np.ndarray, reconstructed: np.ndarray):
        """
        Initialize with original and reconstructed images.
        
        Args:
            original: Original input image
            reconstructed: Reconstructed pixel art
        """
        self.original = original
        self.reconstructed = reconstructed
    
    def create_difference_overlay(self, cell_size: int) -> np.ndarray:
        """
        Create high-contrast difference overlay for objective accuracy validation.
        
        This is the ultimate objective test - visually flags any deviation and helps
        identify exactly where the algorithm is failing.
        
        Args:
            cell_size: Cell size for upscaling reconstructed image
            
        Returns:
            Difference overlay image (magenta on black)
        """
        # Upscale reconstructed image back to original resolution
        original_height, original_width = self.original.shape[:2]
        
        # Use nearest neighbor upscaling to preserve pixel boundaries
        upscaled = cv2.resize(self.reconstructed, 
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
