# Pixel Perfecter

A Python tool for automatically extracting and reconstructing pixel art assets from images. It uses advanced image processing techniques to detect the original pixel scale and segment the artwork into its constituent pixels.

## Features

- Automatic pixel scale detection using FFT (Fast Fourier Transform) analysis
- Smart pixel segmentation using watershed or k-means algorithms
- Support for processing sprite sheets with multiple assets
- Automatic reconstruction of pixel art at the original scale
- Debug visualizations for each processing step

## How It Works

### Scale Detection

The tool uses FFT analysis to detect the original pixel scale of the artwork:

1. Converts image to grayscale and applies light Gaussian blur to reduce noise
2. Performs 2D FFT and analyzes the frequency spectrum
3. Detects frequency peaks corresponding to pixel grid patterns
4. Uses harmonic analysis to identify the fundamental scale from frequency peaks
5. Validates and refines the scale estimate

### Segmentation

Two segmentation methods are supported:

- **Watershed** (Default): Uses morphological operations and watershed segmentation
- **K-means**: Uses color-based clustering for segmentation

### Reconstruction

The tool reconstructs the pixel art by:

1. Calculating the average color for each detected segment
2. Determining the grid position for each segment
3. Reconstructing the final image at the detected scale
4. Applying post-processing to fill any gaps

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alexkorol/pixel-perfecter.git
cd pixel-perfecter
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your pixel art image in the `input` folder

2. Run the tool:
```bash
python src/main.py
```

3. You'll be prompted for:
   - Segmentation method (watershed/kmeans)
   - Whether to use automatic scale detection
   - Number of rows/columns if processing a sprite sheet

4. Processed images will be saved in the `output` folder

### Debug Output

Set `debug_dir` in the code to enable debug visualizations for each processing step, including:
- FFT magnitude spectrum
- Segmentation steps
- Grid detection
- Final reconstruction

## Requirements

See `requirements.txt` for detailed dependencies. Main requirements:
- Python 3.8+
- NumPy
- OpenCV
- scikit-image
- PIL/Pillow

## License

MIT License - See LICENSE file for details