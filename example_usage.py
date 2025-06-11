#!/usr/bin/env python3
"""
Example usage of the grid detection and validation tests.

This script demonstrates how to use the testing methodology to:
1. Detect optimal grid size and offset
2. Validate grid alignment  
3. Test for artifacts and stray pixels
"""

import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from grid_tests import GridDetectionTests, ArtifactTests, run_comprehensive_grid_analysis


def demo_grid_detection(image_path: str):
    """Demonstrate grid detection tests on an image."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    print(f"Analyzing image: {image_path}")
    print("=" * 50)
    
    # Load image
    image = np.array(Image.open(image_path))
    tests = GridDetectionTests(image)
    
    print(f"Image dimensions: {image.shape}")
    
    # 1. Edge projection peak analysis (the breakthrough method)
    print("\n1. Edge Projection Peak Analysis:")
    h_cell, v_cell = tests.edge_projection_peak_analysis()
    print(f"   Detected horizontal cell size: {h_cell}px")
    print(f"   Detected vertical cell size: {v_cell}px")
    
    # 2. Test common cell sizes
    print("\n2. Testing Common Cell Sizes:")
    candidate_sizes = [8, 16, 24, 32, 48, 64]
    
    for cell_size in candidate_sizes:
        # Boundary edge scoring
        edge_score = tests.boundary_edge_scoring(cell_size)
        
        # Intra-block variance
        variance = tests.intra_block_variance(cell_size)
        
        print(f"   {cell_size:2d}px: edge_score={edge_score:6.1f}, variance={variance:6.2f}")
        
        # Save visual grid overlay for top candidates
        if cell_size in [16, 32] or cell_size == h_cell or cell_size == v_cell:
            overlay = tests.visual_grid_overlay(cell_size)
            output_path = f"output/grid_overlay_{cell_size}px.png"
            os.makedirs("output", exist_ok=True)
            Image.fromarray(overlay).save(output_path)
            print(f"      -> Saved grid overlay: {output_path}")
    
    # 3. Recommend best cell size
    print("\n3. Recommendation:")
    if h_cell > 0 and v_cell > 0:
        if h_cell == v_cell:
            recommended = h_cell
            print(f"   Edge projection suggests: {recommended}px (square grid)")
        else:
            recommended = min(h_cell, v_cell)
            print(f"   Edge projection suggests: {recommended}px (rectangular grid {h_cell}x{v_cell}, using smaller)")
    else:
        # Fall back to variance analysis
        variances = {}
        for size in candidate_sizes:
            variances[size] = tests.intra_block_variance(size)
        
        recommended = min(variances.keys(), key=lambda k: variances[k])
        print(f"   Variance analysis suggests: {recommended}px (lowest intra-block variance)")
    
    return recommended


def demo_artifact_detection(original_path: str, processed_path: str, cell_size: int):
    """Demonstrate artifact detection and correction tests."""
    
    if not os.path.exists(original_path) or not os.path.exists(processed_path):
        print("Error: Original or processed image not found")
        return
    
    print(f"\nArtifact Analysis:")
    print("=" * 50)
    
    # Load images
    original = np.array(Image.open(original_path))
    processed = np.array(Image.open(processed_path))
    
    artifact_tests = ArtifactTests(original, processed)
    
    # 1. High-resolution difference overlay (the breakthrough validation method)
    print("1. High-Resolution Difference Overlay:")
    diff_overlay = artifact_tests.high_resolution_difference_overlay(cell_size)
    
    # Count non-zero pixels in overlay
    diff_pixels = np.sum(diff_overlay[:, :, 0] > 0)  # Count magenta pixels
    total_pixels = diff_overlay.shape[0] * diff_overlay.shape[1]
    diff_percentage = (diff_pixels / total_pixels) * 100
    
    print(f"   Difference pixels: {diff_pixels:,} / {total_pixels:,} ({diff_percentage:.2f}%)")
    
    output_path = "output/difference_overlay.png"
    os.makedirs("output", exist_ok=True)
    Image.fromarray(diff_overlay).save(output_path)
    print(f"   -> Saved difference overlay: {output_path}")
    
    # 2. Block-difference stray filter
    print("\n2. Block-Difference Stray Filter:")
    stray_mask = artifact_tests.block_difference_stray_filter(cell_size)
    stray_blocks = np.sum(stray_mask > 0)
    total_blocks = (original.shape[0] // cell_size) * (original.shape[1] // cell_size)
    
    print(f"   Stray blocks detected: {stray_blocks} / {total_blocks}")
    
    # Save stray mask visualization
    stray_viz = np.zeros((stray_mask.shape[0], stray_mask.shape[1], 3), dtype=np.uint8)
    stray_viz[stray_mask > 0] = [255, 255, 0]  # Yellow for stray blocks
    
    output_path = "output/stray_blocks.png"
    Image.fromarray(stray_viz).save(output_path)
    print(f"   -> Saved stray block visualization: {output_path}")
    
    # 3. Logical neighbor majority vote correction
    print("\n3. Logical Neighbor Majority Vote Correction:")
    corrected = artifact_tests.logical_neighbor_majority_vote(cell_size)
    
    output_path = "output/corrected_image.png"
    Image.fromarray(corrected).save(output_path)
    print(f"   -> Saved corrected image: {output_path}")
    
    # Test the correction
    corrected_tests = ArtifactTests(original, corrected)
    corrected_diff = corrected_tests.high_resolution_difference_overlay(cell_size)
    corrected_diff_pixels = np.sum(corrected_diff[:, :, 0] > 0)
    corrected_diff_percentage = (corrected_diff_pixels / total_pixels) * 100
    
    print(f"   After correction: {corrected_diff_pixels:,} / {total_pixels:,} ({corrected_diff_percentage:.2f}%)")
    
    improvement = diff_percentage - corrected_diff_percentage
    print(f"   Improvement: {improvement:.2f} percentage points")


def main():
    """Main demonstration function."""
    print("Pixel Perfecter - Grid Detection Tests")
    print("======================================")
    
    # Check for input images
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"Creating {input_dir}/ directory...")
        os.makedirs(input_dir)
        print("Please place test images in the input/ directory and run again.")
        return
    
    # Look for test images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in input/ directory.")
        print("Please add some pixel art images to test with.")
        return
    
    # Test with first image found
    test_image = os.path.join(input_dir, image_files[0])
    print(f"Using test image: {test_image}")
    
    # Run grid detection
    recommended_cell_size = demo_grid_detection(test_image)
    
    # If we have multiple images, try artifact detection
    if len(image_files) >= 2:
        processed_image = os.path.join(input_dir, image_files[1])
        print(f"\\nTesting artifact detection between:")
        print(f"  Original: {test_image}")
        print(f"  Processed: {processed_image}")
        
        demo_artifact_detection(test_image, processed_image, recommended_cell_size)
    else:
        print("\\nTo test artifact detection, add a second image to input/ directory")
        print("(e.g., a processed/snapped version of the first image)")
    
    print("\\nAll outputs saved to output/ directory")


if __name__ == "__main__":
    main()
