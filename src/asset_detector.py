# asset_detector.py

from PIL import Image
import numpy as np
import os

def load_image(path):
    """Load an image and ensure it's in RGBA format"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert('RGBA')

def crop_to_content(image, padding=0):
    """
    Crop an image to its non-transparent content.
    Returns the cropped image and the crop coordinates.
    """
    np_image = np.array(image)
    alpha = np_image[..., 3]
    
    # Find the bounding box of non-transparent pixels
    coords = np.argwhere(alpha > 0)
    if len(coords) == 0:
        print("Warning: No non-transparent pixels found in image")
        return image, (0, 0, image.width, image.height)
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Add padding
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(image.height, y1 + padding)
    x1 = min(image.width, x1 + padding)
    
    # Crop the image
    cropped = image.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1)

def detect_assets(image, rows, cols, debug_dir=None):
    """
    Split an image into a grid and extract individual assets.
    Returns a list of cropped asset images.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # Save the input image for reference
        image.save(os.path.join(debug_dir, "00_input_spritesheet.png"))
    
    width, height = image.size
    asset_width = width // cols
    asset_height = height // rows
    
    print(f"Grid cell size: {asset_width}x{asset_height} pixels")
    assets = []

    for row in range(rows):
        for col in range(cols):
            # Calculate grid cell boundaries
            left = col * asset_width
            upper = row * asset_height
            right = (col + 1) * asset_width
            lower = (row + 1) * asset_height
            
            # Extract and crop the asset
            grid_cell = image.crop((left, upper, right, lower))
            asset_cropped, crop_coords = crop_to_content(grid_cell)
            
            if debug_dir:
                # Save the grid cell and cropped asset
                grid_cell.save(os.path.join(debug_dir, f"01_grid_cell_{row}_{col}.png"))
                asset_cropped.save(os.path.join(debug_dir, f"02_cropped_asset_{row}_{col}.png"))
                
                # Print crop coordinates for debugging
                print(f"Asset {row}_{col} crop coords: {crop_coords}")
                print(f"Asset {row}_{col} final size: {asset_cropped.size}")
            
            assets.append(asset_cropped)

    return assets