from asset_detector import load_image, detect_assets
from segmentation import segment_image, calculate_segment_colors, calculate_segment_positions
from reconstruction import reconstruct_image, save_image, fill_transparent_gaps
from analysis import estimate_scale_fft
import os
import glob
import numpy as np

def process_image(input_path, output_path, manual_scale=None, method='watershed', rows=1, cols=1):
    """
    Process a single image through the pixel-perfecter pipeline.
    """
    # Create debug output directory
    debug_dir = os.path.join(os.path.dirname(output_path), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load the high-resolution image
    print(f"\nProcessing: {input_path}")
    spritesheet = load_image(input_path)
    
    # Detect and process individual assets from spritesheet
    print(f"Separating spritesheet into {rows}x{cols} grid...")
    assets = detect_assets(spritesheet, rows, cols, debug_dir)
    print(f"Found {len(assets)} assets")
    
    # Process each asset
    for idx, asset in enumerate(assets):
        print(f"\nProcessing asset {idx+1}/{len(assets)}")
        asset_debug_dir = os.path.join(debug_dir, f"asset_{idx}")
        os.makedirs(asset_debug_dir, exist_ok=True)
        
        # Auto-detect scale if not provided
        if manual_scale is None:
            print("Attempting automatic scale detection...")
            scale = estimate_scale_fft(asset, idx, asset_debug_dir)
            if scale is None:
                scale = int(input(f"Could not auto-detect scale for asset {idx+1}. Please enter scale factor manually: "))
        else:
            scale = manual_scale
            
        # Segment the image
        print(f"Segmenting asset using {method} method...")
        label_map = segment_image(asset, scale, method=method, asset_idx=idx, debug_dir=asset_debug_dir)

        # Calculate segment colors and positions
        print("Calculating segment colors...")
        segment_colors = calculate_segment_colors(label_map, asset, method='median')
        
        print("Calculating segment positions...")
        segment_positions = calculate_segment_positions(label_map, scale)

        # Reconstruct the image from segments
        print("Reconstructing final image...")
        reconstructed_image = reconstruct_image(segment_colors, segment_positions)
        
        # Fill any transparent gaps
        print("Applying post-processing to fill transparent gaps...")
        reconstructed_image = fill_transparent_gaps(reconstructed_image)

        # Save the reconstructed asset
        if len(assets) > 1:
            # If multiple assets, save each one separately
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            asset_output_path = os.path.join(
                os.path.dirname(output_path),
                f"{base_name}_asset_{idx}.png"
            )
        else:
            asset_output_path = output_path
            
        save_image(reconstructed_image, asset_output_path)
        print(f"Saved reconstructed asset to: {asset_output_path}")

def main():
    # Get all PNG files in the input directory
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    
    input_files = glob.glob(os.path.join(input_dir, '*.png'))
    
    if not input_files:
        print("No PNG files found in the input directory!")
        return

    # Get user input for parameters
    method = input("Enter segmentation method (watershed/kmeans) [default: watershed]: ").lower() or 'watershed'
    if method not in ['watershed', 'kmeans']:
        print("Invalid method, defaulting to watershed")
        method = 'watershed'
    
    # Ask if user wants to use auto-scale detection
    auto_scale = input("Use automatic scale detection? [Y/n]: ").lower() != 'n'
    manual_scale = None if auto_scale else int(input("Enter the scale factor manually: "))
    
    # Get spritesheet grid dimensions
    rows = int(input("Enter number of rows in spritesheet [default: 2]: ") or "2")
    cols = int(input("Enter number of columns in spritesheet [default: 2]: ") or "2")

    # Process each image
    for input_path in input_files:
        # Create output filename based on input filename
        filename = os.path.basename(input_path)
        output_filename = f"reconstructed_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            process_image(input_path, output_path, manual_scale, method, rows, cols)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == '__main__':
    main()
