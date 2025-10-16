import os
import logging
import argparse
from PIL import Image
import numpy as np
import io

# Import module code
from .asset_detector import detect_assets_in_grid
from .scale_detection import detect_scale
from .segmentation import segment_image
from .reconstruction import reconstruct_image_adaptive
from .pixel_perfecter import process_image as process_with_pixel_perfecter

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
    parser.add_argument('--num_colors', type=int, default=16, help='Target number of colors for quantization')
    # Consider adding blur_size to CLI args if needed later
    # parser.add_argument('--blur_size', type=int, default=5, help='Median blur size (odd number, 0 to disable)')
    return parser

def process_image(image, method='watershed', auto_scale=True, manual_scale=None, rows=2, cols=2, debug_dir=None, blur_size=5, num_colors=16):
    """Process a single image with given parameters

    Args:
        image: Input PIL Image
        method: Segmentation method ('watershed', 'kmeans', or 'pixel_perfecter')
        auto_scale: Whether to auto-detect pixel scale
        manual_scale: Manually specified pixel scale (overrides auto_scale if provided)
        rows: Number of rows in asset grid
        cols: Number of columns in asset grid
        debug_dir: Directory for debug outputs
        blur_size: Size of median blur kernel (odd number, 0 to disable)
        num_colors: Target number of colors for final palette
    """
    # Validate blur_size
    if blur_size < 0 or (blur_size > 0 and blur_size % 2 == 0):
        raise ValueError("Blur size must be an odd number >= 1, or 0 to disable.")

    # Detect assets in grid
    assets = detect_assets_in_grid(image, rows, cols)
    if not assets:
        raise ValueError("No assets found in image")

    # Process each asset
    results = []
    for i, (asset, crop_coords, size) in enumerate(assets):
        # If using pixel_perfecter method, use the new algorithm
        if method == 'pixel_perfecter':
            logging.info(f"Asset {i}: Using pixel_perfecter algorithm")
            result = process_with_pixel_perfecter(
                asset,
                num_colors=num_colors,
                debug_dir=debug_dir,
                asset_idx=i
            )
            reconstructed = result.image
            logging.info(f"Asset {i}: Pixel perfecter created image of size {reconstructed.width}x{reconstructed.height}")
            results.append(reconstructed)
            continue
            
        # Otherwise use the original algorithm
        # Determine scale
        scale = None
        if not auto_scale and manual_scale is not None:
            scale = manual_scale
            logging.info(f"Asset {i}: Using provided manual scale: {scale}")
        elif auto_scale:
            scale = detect_scale(asset)
            if scale:
                logging.info(f"Asset {i}: Using detected scale: {scale}")

        # Handle scale detection failure or default
        if scale is None:
            default_scale = 8
            logging.warning(f"Asset {i}: Scale detection failed or not provided. Defaulting to scale {default_scale}.")
            scale = default_scale

        # Segment the image
        segment_result = segment_image(
            asset,
            scale,
            method=method,
            asset_idx=i,
            debug_dir=debug_dir,
            blur_size=blur_size # Pass blur_size here
        )

        if segment_result is None:
            raise ValueError(f"Segmentation failed for asset {i}")
        labels, colors = segment_result
        logging.info(f"Asset {i}: Segmentation returned {labels.max()} segments and {len(colors)} colors.")
        if labels.max() == 0 or len(colors) == 0:
             logging.warning(f"Asset {i}: Segmentation resulted in zero segments or colors. Reconstruction will likely fail.")


        # Reconstruct using adaptive method
        reconstructed = reconstruct_image_adaptive(
            labels,
            colors,
            num_colors=num_colors,
            debug_dir=debug_dir
        )

        results.append(reconstructed)

    return results[0] if len(results) == 1 else combine_assets(results, rows, cols)

def combine_assets(assets, rows, cols):
    """Combine multiple assets into a single image"""
    # Simple implementation - assumes all assets are same size
    width, height = assets[0].size
    combined = Image.new('RGB', (width * cols, height * rows))

    for i, asset in enumerate(assets):
        row = i // cols
        col = i % cols
        combined.paste(asset, (col * width, row * height))

    return combined

# Updated function signature to include blur_size
def process_file_for_web(input_path, args, method='watershed', auto_scale=True, manual_scale=None, rows=2, cols=2, blur_size=5, num_colors=16):
    """Process a file for web interface"""
    try:
        image = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # Pass blur_size to process_image
    return process_image(
        image,
        method=method,
        auto_scale=auto_scale,
        manual_scale=manual_scale,
        rows=rows,
        cols=cols,
        debug_dir=args.debug_dir,
        blur_size=blur_size, # Pass blur_size here
        num_colors=num_colors
    )

if __name__ == "__main__":
    setup_logging()
    parser = parse_args()
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    # Get list of input files
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not input_files:
        logging.error(f"No image files found in {args.input_dir}")
        exit(1)

    # For each input file
    for input_file in input_files:
        output_path = os.path.join(args.output_dir, f"reconstructed_{input_file}")
        try: # Added try-except block for CLI execution
            result = process_file_for_web(
                os.path.join(args.input_dir, input_file),
                args,
                method='watershed',
                auto_scale=True,
                manual_scale=None,
                rows=2, # Example default
                cols=2, # Example default
                blur_size=5, # Added default blur_size for CLI
                num_colors=args.num_colors
            )
            result.save(output_path)
            logging.info(f"Saved reconstructed image to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to process {input_file}: {e}")
