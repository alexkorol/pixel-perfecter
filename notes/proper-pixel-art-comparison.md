# Hough-Based Grid Detection: Implementation Notes

## Credits and Acknowledgments

This implementation is heavily inspired by **[proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art)** by **Kenneth J. Allen**. The Hough line transform approach, morphological closing technique, and line clustering algorithm were adapted from that excellent project.

Key techniques borrowed from proper-pixel-art:
- Morphological closing to bridge edge gaps
- Hough line transform for direct grid line detection
- Line clustering with median representatives
- Pixel width calculation from line gaps with outlier filtering

## Problem Statement

AI-generated "pixel art" (from tools like GPT-4o, Midjourney, etc.) contains soft halos and anti-aliasing artifacts that make it unsuitable for use as actual pixel art assets. The goal is to reconstruct clean, grid-aligned pixel art from these noisy inputs.

## Original Approach: Autocorrelation (Before)

The original pixel-perfecter used autocorrelation on edge projections to detect grid periodicity:

```
Edge Detection → Project onto axes → Autocorrelation → Find peaks → Select period
```

**Problems:**
1. **Harmonic confusion**: Autocorrelation finds peaks at both fundamental and harmonic frequencies (2x, 0.5x)
2. **Metric refinement backfires**: Optimizing for low reconstruction diff actually favors keeping halos
3. **Weak periodicity**: Noisy edges produce weak/inconsistent autocorrelation signals

**Results**: 32% exact match rate with proper-pixel-art, 53% significant errors

## New Approach: Hough Line Transform (After)

The new implementation uses Hough line detection to find actual grid lines:

```
Edge Detection → Morphological Closing → Hough Lines → Cluster Lines →
Compute Pixel Width from Gaps → Restricted Refinement
```

### Key Components

#### 1. Morphological Closing
```python
def _close_edges(self, edges, kernel_size=8):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```
Bridges small gaps in edge maps to create continuous lines for Hough detection.

#### 2. Hough Line Detection with Angle Filtering
```python
lines = cv2.HoughLinesP(closed, rho=1.0, theta=np.pi/180, threshold=100,
                        minLineLength=50, maxLineGap=10)
# Keep only lines within ±15° of horizontal/vertical
```
Directly detects grid lines rather than inferring periodicity.

#### 3. Line Clustering
```python
def _cluster_lines(self, lines, threshold=4):
    # Merge lines within 4 pixels, return median of each cluster
```
Consolidates multiple detections of the same grid line.

#### 4. Pixel Width from Gaps with Outlier Filtering
```python
def _get_pixel_width_from_gaps(self, lines_x, lines_y, trim_fraction=0.2):
    # Compute gaps between lines
    # Filter out top/bottom 20% as outliers
    # Return median of remaining gaps
```
Robust cell size estimation.

#### 5. Restricted Refinement
When Hough detection is confident (>50% of expected lines detected), restrict the refinement search to ±10% of the detected size. This prevents the metric optimization from choosing wildly different cell sizes.

### Sanity Checks

- **Trivial mesh check**: If ≤3 lines detected per axis, fall back to autocorrelation
- **Cell count check**: If detected cell size would result in <15 cells per dimension, fall back
- **Confidence threshold**: Only trust Hough results with >30% confidence

## Results Comparison

### Test Set: 78 AI-generated pixel art images

| Metric | Before (Autocorr) | After (Hough) | Change |
|--------|-------------------|---------------|--------|
| Exact matches (<5% diff) | 25 (32%) | 31 (40%) | **+24%** |
| Close matches (5-20% diff) | 12 (15%) | 29 (37%) | +147% |
| Significant errors (>20% diff) | 41 (53%) | 18 (23%) | **-56%** |

### Example Improvements

| Image | Before | After | PPA Reference |
|-------|--------|-------|---------------|
| medallion | 9px (wrong) | 13px (correct) | 13px |
| Backpack | 32px | 32px | 32px |
| Sword | 32px | 32px | 32px |

## CLI Usage

```bash
# Default: Hough detection enabled
python -m pixel_perfecter.cli input/*.png -o output/

# Disable Hough (use original autocorrelation)
python -m pixel_perfecter.cli input/*.png -o output/ --no-hough

# Debug mode to see detection details
python -m pixel_perfecter.cli input/*.png -o output/ --debug
```

## Remaining Limitations

Some images still have issues (23% significant errors). These are typically:
- Small pixel art on large canvas (weak edge detection)
- Very fine pixel grids (<8px cells)
- Images with unusual aspect ratios

**Potential future improvements:**
1. Initial 2x upscaling before detection (like proper-pixel-art)
2. Adaptive Hough parameters based on image size
3. Optional color quantization preprocessing

## References

- [proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art) - Kenneth J. Allen
- [OpenCV Hough Lines](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
