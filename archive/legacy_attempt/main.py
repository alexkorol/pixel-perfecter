import os
import sys
import logging
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import module code
from asset_detector import detect_assets_in_grid
from scale_detection import detect_scale
from segmentation import segment_image
from reconstruction import reconstruct_image_adaptive

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Convert high-res pixel art to low-res pixel art')
    parser.add_argument('--input_dir', default='input', help='Directory containing input images')
    parser.add_argument('--output_dir', default='output', help='Directory for output images')
    parser.add_argument('--debug_dir', default='output/debug', help='Directory for debug output')
    return parser.parse_args()

def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)
    
    # Get list of input files
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not input_files:
        logging.error(f"No image files found in {args.input_dir}")
        return
    
    # For each input file
    for input_file in input_files:
        process_file(input_file, args)
        
def process_file(input_file, args):
    """Process a single input file"""
    input_path = os.path.join(args.input_dir, input_file)
    logging.info(f"\nProcessing: {input_path}")
    
    # Load image
    try:
        image = Image.open(input_path)
    except Exception as e:
        logging.error(f"Failed to open {input_path}: {e}")
        return
    
    # Get segmentation method preference
    method = input("Enter segmentation method (watershed/kmeans) [default: watershed]: ").strip().lower()
    if not method:
        method = 'watershed'
    if method not in ['watershed', 'kmeans']:
        logging.error(f"Unknown segmentation method: {method}")
        return
    
    # Determine if we should use automatic scale detection
    auto_scale = input("Use automatic scale detection? [Y/n]: ").strip().lower()
    auto_scale = auto_scale != 'n'
    
    # Ask for grid dimensions
    rows = input("Enter number of rows in spritesheet [default: 2]: ").strip()
    rows = int(rows) if rows else 2
    
    cols = input("Enter number of columns in spritesheet [default: 2]: ").strip()
    cols = int(cols) if cols else 2
    
    # Detect assets in grid
    assets = detect_assets_in_grid(image, rows, cols)
    logging.info(f"Found {len(assets)} assets")
    
    # Process each asset
    for i, (asset, crop_coords, size) in enumerate(assets):
        logging.info(f"\nProcessing asset {i+1}/{len(assets)}")
        
        # Auto-detect or ask for scale
        scale = None
        if auto_scale:
            logging.info("Attempting automatic scale detection...")
            scale = detect_scale(asset)
            if scale:
                logging.info(f"Detected scale: {scale}")
            else:
                logging.warning("Automatic scale detection failed")
        
        if not scale:
            scale_input = input("Enter pixel scale (e.g., 8 for 8x8 upscaled pixels) [default: 8]: ").strip()
            scale = int(scale_input) if scale_input else 8
        
        # Create asset-specific debug directory
        asset_debug_dir = os.path.join(args.debug_dir, f"asset_{i}")
        os.makedirs(asset_debug_dir, exist_ok=True)
        
        # Segment the image
        segment_result = segment_image(
            asset,
            scale,
            method=method,
            asset_idx=i,
            debug_dir=asset_debug_dir
        )
        
        if segment_result is None:
            logging.error(f"Segmentation failed for asset {i}")
            continue
        
        labels, colors = segment_result
        
        # Reconstruct using adaptive method
        reconstructed = reconstruct_image_adaptive(
            labels,
            colors,
            debug_dir=asset_debug_dir
        )
        
        # Save the reconstructed image
        base_name = os.path.splitext(input_file)[0]
        output_path = os.path.join(args.output_dir, f"reconstructed_{base_name}_asset_{i}.png")
        reconstructed.save(output_path)
        logging.info(f"Saved reconstructed asset to: {output_path}")

if __name__ == "__main__":
    main()