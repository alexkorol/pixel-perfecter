#!/usr/bin/env python3
"""
Test all simulated pixel art images to find optimal grid parameters.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from PIL import Image
from grid_tests import GridDetectionTests

def test_image(image_path, name):
    """Test a single image."""
    print(f"\n{'='*50}")
    print(f"TESTING: {name}")
    print(f"{'='*50}")
    
    image = np.array(Image.open(image_path))
    tests = GridDetectionTests(image)
    
    print(f"Image shape: {image.shape}")
    
    # Test common pixel art grid sizes
    candidate_sizes = [16, 24, 32, 40, 48, 64]
    
    results = []
    for cell_size in candidate_sizes:
        edge_score = tests.boundary_edge_scoring(cell_size)
        variance = tests.intra_block_variance(cell_size)
        results.append((cell_size, edge_score, variance))
        
        # Save overlay
        overlay = tests.visual_grid_overlay(cell_size)
        os.makedirs("output", exist_ok=True)
        Image.fromarray(overlay).save(f"output/{name}_grid_{cell_size}px.png")
    
    # Show results
    print("Size | Edge Score | Variance")
    print("-----|------------|----------")
    best_edge_score = max(r[1] for r in results)
    best_variance = min(r[2] for r in results)
    
    for cell_size, edge_score, variance in results:
        marker = ""
        if edge_score == best_edge_score:
            marker += " <-- BEST EDGE"
        if variance == best_variance:
            marker += " <-- BEST VARIANCE"
        print(f"{cell_size:3d}px | {edge_score:8.1f} | {variance:8.2f}{marker}")
    
    # Edge projection
    h_cell, v_cell = tests.edge_projection_peak_analysis(edge_threshold=100)
    print(f"Edge projection detected: {h_cell}x{v_cell}")
    
    return results

def main():
    """Test all images."""
    print("PIXEL ART GRID ANALYSIS")
    print("=======================")
    
    # Map of files to short names
    image_map = {
        '20250610_1843_Red Triangle Pixel Art_simple_compose_01jxeafmtye8v8zpqyn290t9nb.png': 'triangle',
        '20250610_2049_Green Smiley Pixel Art_simple_compose_01jxehphg7fjhrc4gy66858g3k.png': 'smiley', 
        '20250610_2037_Pixel Gold Star_simple_compose_01jxegz6xsfk6bc9dng8a92zhs.png': 'star',
        '20250329_2258_Flytrap Pollen Close-Up_remix_01jqjt2qw9fmvaj968ftkvmvfx.png': 'flytrap'
    }
    
    all_results = {}
    
    for filename, name in image_map.items():
        image_path = os.path.join('input', filename)
        if os.path.exists(image_path):
            results = test_image(image_path, name)
            all_results[name] = results
        else:
            print(f"Warning: {filename} not found")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for name, results in all_results.items():
        best_edge = max(results, key=lambda x: x[1])
        best_variance = min(results, key=lambda x: x[2])
        
        print(f"{name.upper()}:")
        print(f"  Best edge score: {best_edge[0]}px (score: {best_edge[1]:.1f})")
        print(f"  Best variance: {best_variance[0]}px (var: {best_variance[2]:.2f})")
        
        if best_edge[0] == best_variance[0]:
            print(f"  --> RECOMMENDED: {best_edge[0]}px (both metrics agree)")
        else:
            print(f"  --> Consider: {best_edge[0]}px or {best_variance[0]}px")

if __name__ == "__main__":
    main()
