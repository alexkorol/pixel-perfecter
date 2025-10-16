#!/usr/bin/env python3
"""
Analyze each simulated pixel art image individually to understand their structure.
"""

import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from grid_tests import GridDetectionTests


def analyze_single_image(image_path: str):
    """Analyze a single image in detail."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Load image
    image = np.array(Image.open(image_path))
    tests = GridDetectionTests(image)
    
    print(f"Image dimensions: {image.shape}")
    
    # 1. Edge projection analysis with different thresholds
    print("\n1. Edge Projection Analysis (different thresholds):")
    for threshold in [30, 50, 100, 150]:
        h_cell, v_cell = tests.edge_projection_peak_analysis(edge_threshold=threshold)
        print(f"   Threshold {threshold:3d}: horizontal={h_cell:2d}px, vertical={v_cell:2d}px")
    
    # 2. Test a wider range of cell sizes
    print("\n2. Comprehensive Cell Size Testing:")
    candidate_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
    
    results = []
    for cell_size in candidate_sizes:
        edge_score = tests.boundary_edge_scoring(cell_size)
        variance = tests.intra_block_variance(cell_size)
        results.append((cell_size, edge_score, variance))
        
        # Create grid overlay for visual inspection
        overlay = tests.visual_grid_overlay(cell_size)
        output_path = f"output/{os.path.basename(image_path)[:-4]}_grid_{cell_size}px.png"
        os.makedirs("output", exist_ok=True)
        Image.fromarray(overlay).save(output_path)
    
    # Sort by edge score (descending) and variance (ascending)
    print("   Size | Edge Score | Variance | Notes")
    print("   -----|------------|----------|------")
    
    for cell_size, edge_score, variance in results:
        note = ""
        if edge_score == max(r[1] for r in results):
            note += "BEST_EDGE "
        if variance == min(r[2] for r in results):
            note += "BEST_VAR "
        
        print(f"   {cell_size:3d}px | {edge_score:8.1f} | {variance:7.2f} | {note}")
    
    # 3. Find best candidates
    best_edge = max(results, key=lambda x: x[1])
    best_variance = min(results, key=lambda x: x[2])
    
    print(f"\n3. Best Candidates:")
    print(f"   Highest edge score: {best_edge[0]}px (score: {best_edge[1]:.1f})")
    print(f"   Lowest variance: {best_variance[0]}px (variance: {best_variance[2]:.2f})")
    
    # 4. Create overlays for best candidates
    print(f"\n4. Creating detailed overlays for best candidates...")
    for label, (cell_size, _, _) in [("edge", best_edge), ("variance", best_variance)]:
        overlay = tests.visual_grid_overlay(cell_size, color=(255, 0, 255))  # Magenta
        output_path = f"output/{os.path.basename(image_path)[:-4]}_BEST_{label}_{cell_size}px.png"
        Image.fromarray(overlay).save(output_path)
        print(f"   -> Saved {label} overlay: {output_path}")


def main():
    """Analyze all images in the input directory."""
    print("Detailed Pixel Art Analysis")
    print("===========================")
    
    input_dir = "input"
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in input/ directory.")
        return
    
    print(f"Found {len(image_files)} images to analyze:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img}")
    
    # Analyze each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        analyze_single_image(image_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("All grid overlays saved to output/ directory")
    print("Look for files ending with 'BEST_edge_XXpx.png' and 'BEST_variance_XXpx.png'")
    print("These show the most promising grid alignments for each image.")


if __name__ == "__main__":
    main()
