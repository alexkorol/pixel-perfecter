import numpy as np
from PIL import Image

def fill_transparent_gaps(image):
    """
    Fill isolated transparent pixels with colors from their neighbors.
    
    Args:
        image: PIL Image in RGBA format
    Returns:
        PIL Image with filled transparent pixels
    """
    # Convert to numpy array
    arr = np.array(image)
    h, w = arr.shape[:2]
    
    # Find transparent pixels
    transparent = arr[..., 3] == 0
    
    # Skip if no transparent pixels
    if not np.any(transparent):
        return image
    
    # Create array for the filled result
    filled = arr.copy()
    
    # Define neighbor offsets (8-connected neighbors)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Count how many pixels we fill
    filled_count = 0
    
    # Find coordinates of transparent pixels
    y_coords, x_coords = np.where(transparent)
    
    for y, x in zip(y_coords, x_coords):
        neighbor_colors = []
        
        # Check all neighbors
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            
            # Check bounds and transparency
            if (0 <= ny < h and 0 <= nx < w and 
                arr[ny, nx, 3] > 0):
                neighbor_colors.append(arr[ny, nx])
        
        # If we found opaque neighbors, fill the pixel
        if neighbor_colors:
            # Convert to numpy array for easier calculation
            neighbor_colors = np.array(neighbor_colors)
            
            # Use median color of neighbors
            filled[y, x] = np.median(neighbor_colors, axis=0).astype(np.uint8)
            filled_count += 1
    
    if filled_count > 0:
        print(f"Filled {filled_count} transparent pixels")
    
    return Image.fromarray(filled)

def reconstruct_image(segment_colors, segment_positions):
    """
    Reconstructs the final pixel art image from segment colors and positions.
    """
    if not segment_colors or not segment_positions:
        raise ValueError("Empty segment colors or positions")

    # Determine output dimensions
    max_col = max(pos[0] for pos in segment_positions.values())
    max_row = max(pos[1] for pos in segment_positions.values())
    width, height = max_col + 1, max_row + 1
    print(f"Determined output dimensions: {width}x{height} pixels")
    
    # Create output array (initialize as transparent)
    output_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Place colors at their target positions
    debug_count = 0
    print(f"Processing {len(segment_colors)} segments...")
    
    for segment_id, color in segment_colors.items():
        if segment_id in segment_positions:
            col, row = segment_positions[segment_id]
            if 0 <= row < height and 0 <= col < width:
                output_array[row, col] = color
                # Print first few segments for debugging
                if debug_count < 5:
                    print(f"Placed color {tuple(color)} for segment {segment_id} at ({col}, {row})")
                    debug_count += 1
                elif debug_count == 5:
                    print("... (more segments follow)")
                    debug_count += 1
    
    pixels_placed = np.sum(output_array[..., 3] > 0)
    print(f"Placed {pixels_placed} non-transparent pixels in output")
    
    # Convert to PIL Image
    output_image = Image.fromarray(output_array, 'RGBA')
    return output_image

def save_image(image, path):
    """
    Save the reconstructed image to a file.
    """
    print(f"Saving image of size {image.size} to {path}")
    image.save(path, 'PNG')
