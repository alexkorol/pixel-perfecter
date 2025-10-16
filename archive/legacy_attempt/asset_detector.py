# asset_detector.py

from PIL import Image
from typing import List, Tuple
import numpy as np

def detect_assets_in_grid(
    image: Image.Image,
    rows: int = 2,
    cols: int = 2
) -> List[Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int]]]:
    """
    Extract individual assets from a spritesheet.
    
    Args:
        image: PIL Image of the spritesheet
        rows: Number of rows in the spritesheet
        cols: Number of columns in the spritesheet
        
    Returns:
        List of tuples containing:
        - PIL Image: The extracted asset
        - Tuple[int, int, int, int]: Crop coordinates (left, top, right, bottom)
        - Tuple[int, int]: Final size of the asset (width, height)
    """
    print(f"Separating spritesheet into {rows}x{cols} grid...")
    
    # Calculate cell dimensions
    cell_width = image.width // cols
    cell_height = image.height // rows
    print(f"Grid cell size: {cell_width}x{cell_height} pixels")
    
    assets = []
    for row in range(rows):
        for col in range(cols):
            # Calculate crop coordinates
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            print(f"Asset {row}_{col} crop coords: ({left}, {top}, {right}, {bottom})")
            
            # Extract and crop the asset
            asset = image.crop((left, top, right, bottom))
            
            # Store original crop coordinates
            crop_coords = (left, top, right, bottom)
            
            # Trim transparent borders if they exist
            if asset.mode == 'RGBA':
                # Get alpha channel
                alpha = np.array(asset.getchannel('A'))
                # Find non-transparent pixels
                non_transparent = np.where(alpha > 0)
                if len(non_transparent[0]) > 0:  # If there are non-transparent pixels
                    # Get bounding box
                    top_trim = non_transparent[0].min()
                    bottom_trim = non_transparent[0].max() + 1
                    left_trim = non_transparent[1].min()
                    right_trim = non_transparent[1].max() + 1
                    # Crop to content
                    asset = asset.crop((left_trim, top_trim, right_trim, bottom_trim))
            
            print(f"Asset {row}_{col} final size: {asset.size}")
            assets.append((asset, crop_coords, asset.size))
    
    return assets