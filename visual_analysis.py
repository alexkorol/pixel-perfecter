#!/usr/bin/env python3
"""
Create a comprehensive analysis with visual grid overlays for manual inspection.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from PIL import Image
from grid_tests import GridDetectionTests

def analyze_image_with_visuals(image_name, sizes_to_test):
    """Create grid overlays for visual inspection."""
    
    image_path = f'input/{image_name}'
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found")
        return
    
    print(f"\nAnalyzing: {image_name}")
    print("="*50)
    
    image = np.array(Image.open(image_path))
    tests = GridDetectionTests(image)
    
    # Edge projection analysis
    h_cell, v_cell = tests.edge_projection_peak_analysis(edge_threshold=100)
    print(f"Edge projection detected: {h_cell}x{v_cell}")
    
    print(f"\nTesting grid sizes: {sizes_to_test}")
    print("Size | Edge Score | Variance")
    print("-----|------------|----------")
    
    os.makedirs("output", exist_ok=True)
    
    results = []
    for size in sizes_to_test:
        edge_score = tests.boundary_edge_scoring(size)
        variance = tests.intra_block_variance(size)
        results.append((size, edge_score, variance))
        
        print(f"{size:3d}px | {edge_score:8.1f} | {variance:8.2f}")
        
        # Create visual overlay
        overlay = tests.visual_grid_overlay(size, color=(255, 0, 255))  # Magenta grid
        
        # Save with descriptive name
        short_name = image_name.split('_')[2].lower()  # Extract key part of name
        output_path = f"output/{short_name}_{size:02d}px_grid.png"
        Image.fromarray(overlay).save(output_path)
    
    # Find candidates
    best_variance = min(results, key=lambda x: x[2])
    best_edge = max(results, key=lambda x: x[1])
    
    print(f"\nBest variance: {best_variance[0]}px (var: {best_variance[2]:.2f})")
    print(f"Best edge score: {best_edge[0]}px (score: {best_edge[1]:.1f})")
    
    if h_cell in sizes_to_test:
        h_result = next((r for r in results if r[0] == h_cell), None)
        if h_result:
            print(f"Edge proj H ({h_cell}px): score={h_result[1]:.1f}, var={h_result[2]:.2f}")
    
    print(f"Visual overlays saved to output/ with pattern: {short_name}_XXpx_grid.png")

def main():
    print("COMPREHENSIVE VISUAL GRID ANALYSIS")
    print("==================================")
    
    # Define images and their likely grid size ranges
    test_cases = [
        ("20250610_1843_Red Triangle Pixel Art_simple_compose_01jxeafmtye8v8zpqyn290t9nb.png", 
         [8, 12, 16, 20, 24, 28, 32, 40, 48, 64]),
        
        ("20250610_2049_Green Smiley Pixel Art_simple_compose_01jxehphg7fjhrc4gy66858g3k.png",
         [8, 12, 16, 20, 24, 28, 32, 40, 48]),
         
        ("20250610_2037_Pixel Gold Star_simple_compose_01jxegz6xsfk6bc9dng8a92zhs.png",
         [8, 12, 16, 20, 24, 32, 40, 48]),
         
        ("20250329_2258_Flytrap Pollen Close-Up_remix_01jqjt2qw9fmvaj968ftkvmvfx.png",
         [12, 16, 20, 24, 28, 32, 40, 48])
    ]
    
    for image_name, sizes in test_cases:
        analyze_image_with_visuals(image_name, sizes)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("Check the output/ directory for grid overlay images.")
    print("Look for overlays where the grid lines align well with the image structure.")
    print("File naming: [triangle/smiley/star/flytrap]_XXpx_grid.png")

if __name__ == "__main__":
    main()
