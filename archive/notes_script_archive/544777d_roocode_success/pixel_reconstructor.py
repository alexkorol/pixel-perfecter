"""
Unified Pixel Art Reconstruction Pipeline

This module contains the consolidated, optimized pixel art reconstruction logic.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import os


class PixelArtReconstructor:
    def debug_grid_detection(self, save_dir=None):
        """
        Save debug plots of edge projections and autocorrelations for this image.
        """
        import matplotlib.pyplot as plt
        import os
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
        h_proj = np.sum(edges, axis=0)
        v_proj = np.sum(edges, axis=1)
        h_autocorr = np.correlate(h_proj, h_proj, mode='full')[len(h_proj)-1:]
        v_autocorr = np.correlate(v_proj, v_proj, mode='full')[len(v_proj)-1:]
        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(h_proj)
        axs[0, 0].set_title('Horizontal Edge Projection')
        axs[0, 1].plot(h_autocorr)
        axs[0, 1].set_title('Horizontal Autocorrelation')
        axs[1, 0].plot(v_proj)
        axs[1, 0].set_title('Vertical Edge Projection')
        axs[1, 1].plot(v_autocorr)
        axs[1, 1].set_title('Vertical Autocorrelation')
        fig.suptitle(f'Grid Debug: {os.path.basename(self.image_path)}')
        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, os.path.basename(self.image_path) + '_grid_debug.png')
            plt.savefig(plot_path)
            plt.close(fig)
        else:
            plt.show()

    """
    Consolidated pixel art reconstruction pipeline.
    
    Implements a three-step process:
    1. Grid parameter detection using edge projection and intra-block variance
    2. Grid snapping using modal color within each cell
    3. Artifact refinement using neighbor majority vote
    """
    
    def __init__(self, image_path: str, debug=False):
        """
        Initialize with input image.
        
        Args:
            image_path: Path to input image file
            debug: Enable debug logging
        """
        self.image_path = image_path
        self.image = None
        self.cell_size = None
        self.offset = None
        self.grid_result = None
        self.debug = debug
        
    def run(self) -> np.ndarray:
        """
        Execute the grid detection and empirical pixel art reconstruction pipeline.
        Returns:
            Reconstructed pixel art image (empirically determined grid)
        """
        print(f"Processing {self.image_path}")
        self.image = self._load_image()
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        print("Step 1: Detecting grid parameters (edge-projection peak analysis)...")
        self.cell_size, self.offset = self._detect_grid_parameters()
        print(f"  Detected cell size: {self.cell_size}, offset: {self.offset}")

        print("Step 2: Empirical pixel art reconstruction...")
        reconstructed = self._empirical_pixel_reconstruction()
        print("Done!")
        return reconstructed

    def _empirical_pixel_reconstruction(self) -> np.ndarray:
        """
        Reconstructs the image by mapping each detected grid cell to a single output pixel (modal color).
        Returns:
            Pixel art image of shape (num_cells_y, num_cells_x, 3)
        """
        height, width, channels = self.image.shape
        x_offset, y_offset = self.offset
        cell_size = self.cell_size

        # Compute number of cells in each direction
        num_cells_x = (width - x_offset) // cell_size
        num_cells_y = (height - y_offset) // cell_size

        out_img = np.zeros((num_cells_y, num_cells_x, 3), dtype=np.uint8)

        for i in range(num_cells_y):
            for j in range(num_cells_x):
                y0 = y_offset + i * cell_size
                x0 = x_offset + j * cell_size
                y1 = min(y0 + cell_size, height)
                x1 = min(x0 + cell_size, width)
                cell = self.image[y0:y1, x0:x1]
                if cell.size > 0:
                    modal_color = self._find_modal_color(cell)
                    out_img[i, j] = modal_color
        return out_img
        
    def _load_image(self) -> Optional[np.ndarray]:
        """Load and convert image to RGB numpy array."""
        try:
            with Image.open(self.image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
            
    def _detect_grid_parameters(self) -> Tuple[int, Tuple[int, int]]:
        """
        Detect grid cell size and offset using edge projection analysis.
        
        Returns:
            Tuple of (cell_size, (x_offset, y_offset))
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find dominant cell size using projection analysis
        cell_size = self._find_dominant_period(edges)
        
        # Find best offset for this cell size
        offset = self._find_best_offset(edges, cell_size)
        
        return cell_size, offset
        
    def _find_dominant_period(self, edges: np.ndarray) -> int:
        """
        Find dominant repeating pattern in edge projections.
        
        Args:
            edges: Binary edge image
            
        Returns:
            Estimated cell size in pixels
        """
        height, width = edges.shape
        
        # Project edges onto both axes
        h_projection = np.sum(edges, axis=0)  # Vertical edges
        v_projection = np.sum(edges, axis=1)  # Horizontal edges
        
        if self.debug:
            print(f"DEBUG: Image dimensions: {width}x{height}")
            print(f"DEBUG: Edge projection sums - Horizontal: {np.sum(h_projection)}, Vertical: {np.sum(v_projection)}")
        
        # Find periods in projections using improved autocorrelation
        h_periods, h_strengths = self._find_period_autocorr(h_projection, "horizontal")
        v_periods, v_strengths = self._find_period_autocorr(v_projection, "vertical")
        
        if self.debug:
            print(f"DEBUG: Horizontal periods and strengths: {list(zip(h_periods, h_strengths))}")
            print(f"DEBUG: Vertical periods and strengths: {list(zip(v_periods, v_strengths))}")
        
        # Combine and analyze periods from both axes
        selected_period = self._select_best_period(h_periods, h_strengths, v_periods, v_strengths, width, height)
        
        if selected_period > 0:
            if self.debug:
                print(f"DEBUG: Selected period: {selected_period}")
            return selected_period
        else:
            # Fallback: try variance-based detection
            if self.debug:
                print("DEBUG: No valid periods found, falling back to variance-based detection")
            return self._detect_cell_size_variance()
            
    def _select_best_period(self, h_periods, h_strengths, v_periods, v_strengths, width, height):
        """
        Select the best period from horizontal and vertical candidates using multiple criteria.
        
        Args:
            h_periods: List of horizontal period candidates
            h_strengths: Strengths of horizontal period candidates
            v_periods: List of vertical period candidates
            v_strengths: Strengths of vertical period candidates
            width: Image width
            height: Image height
            
        Returns:
            Best period estimate
        """
        max_test_size = min(64, width//4, height//4)
        min_test_size = 4
        
        # Filter periods to reasonable range
        h_valid = [(p, s) for p, s in zip(h_periods, h_strengths) if min_test_size <= p <= max_test_size]
        v_valid = [(p, s) for p, s in zip(v_periods, v_strengths) if min_test_size <= p <= max_test_size]
        
        if self.debug:
            print(f"DEBUG: Valid horizontal periods: {h_valid}")
            print(f"DEBUG: Valid vertical periods: {v_valid}")
        
        # If no valid periods, return 0 to trigger fallback
        if not h_valid and not v_valid:
            return 0
        
        # Check for common periods between horizontal and vertical (high confidence)
        # This addresses the "Common Period Detection" requirement - prioritize this first
        
        # First, check for exact matches
        exact_common_periods = []
        for p in set([p for p, _ in h_valid]).intersection([p for p, _ in v_valid]):
            h_strength = next((s for period, s in h_valid if period == p), 0)
            v_strength = next((s for period, s in v_valid if period == p), 0)
            combined_strength = h_strength + v_strength
            exact_common_periods.append((p, combined_strength))
            if self.debug:
                print(f"DEBUG: Found exact common period: {p} with strength {combined_strength:.3f}")
        
        # Then, check for approximate matches (allowing small differences)
        approx_common_periods = []
        for h_p, h_s in h_valid:
            for v_p, v_s in v_valid:
                # Skip exact matches as they're already handled
                if h_p == v_p:
                    continue
                    
                # Allow periods to differ by up to 5% of the smaller period
                tolerance = min(h_p, v_p) * 0.05
                if abs(h_p - v_p) <= tolerance:
                    # Use the average of the two periods
                    avg_period = round((h_p + v_p) / 2)
                    # Calculate combined strength
                    combined_strength = h_s + v_s
                    approx_common_periods.append((avg_period, combined_strength))
                    if self.debug:
                        print(f"DEBUG: Found approximate common period: {avg_period} (h:{h_p}, v:{v_p}) with strength {combined_strength:.3f}")
        
        # Combine exact and approximate matches and select the best one
        all_common_periods = exact_common_periods + approx_common_periods
        
        if all_common_periods:
            # Sort by strength (descending)
            sorted_common = sorted(all_common_periods, key=lambda x: x[1], reverse=True)
            
            # For smiley and star images, we want to prioritize smaller periods when strengths are close
            # This helps with the specific case where both 32 and 64 are detected with similar strengths
            best_period = sorted_common[0][0]
            best_strength = sorted_common[0][1]
            
            # Check if there's a smaller period with similar strength (within 5%)
            for period, strength in sorted_common:
                if period < best_period and strength >= best_strength * 0.95:
                    best_period = period
                    best_strength = strength
                    if self.debug:
                        print(f"DEBUG: Selected smaller period {best_period} with similar strength {best_strength:.3f}")
                    break
            
            if self.debug:
                print(f"DEBUG: Selected best common period: {best_period} with strength {best_strength:.3f}")
            return best_period
        
        # Global peak analysis - combine all peaks for analysis
        all_valid = h_valid + v_valid
        
        # Check for harmonic relationships across all periods
        # This addresses the "Harmonic Relationship Analysis" requirement
        harmonics = self._check_harmonics(all_valid)
        if harmonics:
            if self.debug:
                print(f"DEBUG: Detected harmonic relationship across all periods: {harmonics}")
            # Use the fundamental period (first harmonic)
            fundamental_period = harmonics[0]
            if self.debug:
                print(f"DEBUG: Selected fundamental period from harmonics: {fundamental_period}")
            return fundamental_period
        
        # No common periods or harmonics, use the strongest period from either axis
        # This addresses the "Strength-Based Selection" requirement
        if all_valid:
            # Sort by strength (descending)
            sorted_periods = sorted(all_valid, key=lambda x: x[1], reverse=True)
            best_period = sorted_periods[0][0]
            
            if self.debug:
                print(f"DEBUG: Selected strongest period: {best_period} with strength {sorted_periods[0][1]:.3f}")
            return best_period
            
        # Shouldn't reach here, but just in case
        return 0
        
    def _check_harmonics(self, periods):
        """
        Check if the detected periods form harmonic relationships.
        
        Args:
            periods: List of (period, strength) tuples
            
        Returns:
            List of periods in harmonic relationship, or None
        """
        if len(periods) < 2:
            return None
        
        # Sort periods by value (ascending)
        sorted_periods = sorted(periods, key=lambda x: x[0])
        period_values = [p for p, _ in sorted_periods]
        period_strengths = [s for _, s in sorted_periods]
        
        # Group periods by potential harmonic relationships
        harmonic_groups = []
        
        # For each potential fundamental period
        for i, p1 in enumerate(period_values):
            if p1 < 8:  # Skip very small periods as they're unlikely to be fundamental
                continue
                
            harmonics = [p1]
            strengths = [period_strengths[i]]
            
            # Check which other periods might be harmonics of this one
            for j, p2 in enumerate(period_values):
                if i == j:
                    continue  # Skip self
                
                # Check if p2 is approximately a multiple of p1
                if p2 > p1:
                    ratio = p2 / p1
                    # Check if ratio is close to an integer
                    nearest_int = round(ratio)
                    # More strict criteria for harmonic relationship
                    if abs(ratio - nearest_int) < 0.1 and nearest_int > 1 and nearest_int <= 4:  # Allow 10% error, max 4x multiple
                        harmonics.append(p2)
                        strengths.append(period_strengths[j])
            
            # If we found at least one harmonic relationship
            if len(harmonics) > 1:
                # Calculate a score based on number of harmonics and their strengths
                score = len(harmonics) * sum(strengths)
                harmonic_groups.append((harmonics, score))
        
        # If we found harmonic groups, return the one with the highest score
        if harmonic_groups:
            best_group = max(harmonic_groups, key=lambda x: x[1])
            if self.debug:
                print(f"DEBUG: Best harmonic group: {best_group[0]} with score {best_group[1]:.3f}")
            return best_group[0]
                
        return None
            
    def _find_period_autocorr(self, signal: np.ndarray, direction: str = ""):
        """
        Find dominant periods in 1D signal using autocorrelation.
        
        Args:
            signal: 1D signal array
            direction: Direction label for debug output ("horizontal" or "vertical")
            
        Returns:
            Tuple of (periods, strengths) - lists of period candidates and their strengths
        """
        if len(signal) < 8:
            if self.debug:
                print(f"DEBUG: {direction} signal too short (<8), returning empty results")
            return [], []  # Signal too short
            
        # Compute autocorrelation
        correlation = np.correlate(signal, signal, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        if self.debug:
            print(f"DEBUG: {direction} autocorrelation shape: {correlation.shape}")
            print(f"DEBUG: {direction} autocorrelation[0] (zero lag): {correlation[0]}")
            # Print a few key correlation values
            sample_lags = [4, 8, 16, 32]
            sample_lags = [lag for lag in sample_lags if lag < len(correlation)]
            for lag in sample_lags:
                if lag < len(correlation):
                    ratio = correlation[lag] / correlation[0] if correlation[0] > 0 else 0
                    print(f"DEBUG: {direction} autocorrelation at lag {lag}: {correlation[lag]} (ratio to zero lag: {ratio:.3f})")
        
        # Find all peaks in autocorrelation
        peak_periods = []
        peak_strengths = []
        
        # Use a lower threshold to capture more potential peaks for images with weaker signals
        threshold = 0.4  # Lower threshold as specified in requirements
        
        # Minimum peak distance (to avoid detecting very close peaks)
        min_peak_distance = 3
        
        # Find all peaks using a more robust approach
        for lag in range(4, min(len(correlation), len(signal)//4)):
            # Check if this point is a local maximum
            is_peak = (lag > 0 and correlation[lag] > correlation[lag-1] and
                      lag < len(correlation)-1 and correlation[lag] > correlation[lag+1])
            
            if is_peak:
                # Calculate strength as ratio to zero lag
                strength = correlation[lag] / correlation[0] if correlation[0] > 0 else 0
                
                # Only consider peaks above threshold
                if strength > threshold:
                    # Check if this peak is far enough from previously detected peaks
                    if not peak_periods or min(abs(lag - p) for p in peak_periods) >= min_peak_distance:
                        peak_periods.append(lag)
                        peak_strengths.append(strength)
                        
                        if self.debug:
                            print(f"DEBUG: {direction} Found peak at lag {lag} with strength {strength:.3f}")
        
        if self.debug:
            if peak_periods:
                print(f"DEBUG: {direction} All peaks found: {list(zip(peak_periods, peak_strengths))}")
            else:
                print(f"DEBUG: {direction} No peaks found")
                
        return peak_periods, peak_strengths
        
    def _detect_cell_size_variance(self) -> int:
        """
        Improved fallback method: detect cell size using intra-block variance.
        
        Returns:
            Estimated cell size
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # Test cell sizes from 4 to min(64, width//4, height//4)
        max_test_size = min(64, width//4, height//4)
        
        if self.debug:
            print(f"DEBUG: Variance-based detection testing sizes from 4 to {max_test_size}")
            
        variance_results = []
        edge_alignment_scores = []
        
        # Detect edges for alignment scoring
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
        
        for size in range(4, max_test_size + 1):
            # Calculate variance-based score
            total_variance = 0
            block_count = 0
            
            # Sample blocks across the image
            for y in range(0, height - size, size):
                for x in range(0, width - size, size):
                    block = gray[y:y+size, x:x+size]
                    variance = np.var(block)
                    total_variance += variance
                    block_count += 1
                    
            if block_count > 0:
                avg_variance = total_variance / block_count
                variance_results.append((size, avg_variance))
            
            # Calculate edge alignment score for this size
            alignment_score = 0
            for offset_x in range(size):
                for offset_y in range(size):
                    score = self._score_grid_alignment(edges, size, (offset_x, offset_y))
                    alignment_score = max(alignment_score, score)
            
            edge_alignment_scores.append((size, alignment_score))
        
        if not variance_results:
            if self.debug:
                print("DEBUG: No variance results, analyzing image properties to determine best size")
                
            # Analyze image dimensions for common divisors instead of using fixed fallback
            common_divisors = []
            for i in range(4, min(32, min(width, height)//4)):
                if width % i < i//4 and height % i < i//4:  # Allow some tolerance
                    common_divisors.append(i)
            
            if common_divisors:
                best_size = max(common_divisors)  # Prefer larger divisors
                if self.debug:
                    print(f"DEBUG: Selected size {best_size} based on image dimensions")
                return best_size
            
            # If no good divisors found, use edge detection to estimate
            if edge_alignment_scores:
                best_edge_size = max(edge_alignment_scores, key=lambda x: x[1])[0]
                if self.debug:
                    print(f"DEBUG: Selected size {best_edge_size} based on edge alignment")
                return best_edge_size
                
            # Last resort - analyze frequency domain
            # This is a simplified approach - in a real implementation, we might use FFT
            row_diff = np.diff(gray, axis=1)
            col_diff = np.diff(gray, axis=0)
            
            row_periods = []
            col_periods = []
            
            # Find runs of similar values
            for i in range(min(100, height)):
                run_lengths = []
                current_run = 1
                for j in range(1, width-1):
                    if abs(row_diff[i, j] - row_diff[i, j-1]) < 10:
                        current_run += 1
                    else:
                        if current_run > 3:
                            run_lengths.append(current_run)
                        current_run = 1
                if current_run > 3:
                    run_lengths.append(current_run)
                if run_lengths:
                    row_periods.append(np.median(run_lengths))
            
            for j in range(min(100, width)):
                run_lengths = []
                current_run = 1
                for i in range(1, height-1):
                    if abs(col_diff[i, j] - col_diff[i-1, j]) < 10:
                        current_run += 1
                    else:
                        if current_run > 3:
                            run_lengths.append(current_run)
                        current_run = 1
                if current_run > 3:
                    run_lengths.append(current_run)
                if run_lengths:
                    col_periods.append(np.median(run_lengths))
            
            if row_periods or col_periods:
                all_periods = row_periods + col_periods
                if all_periods:
                    best_size = max(4, int(np.median(all_periods)))
                    if self.debug:
                        print(f"DEBUG: Selected size {best_size} based on run length analysis")
                    return best_size
            
            # If all else fails, use a more reasonable default based on image size
            best_size = max(4, min(16, min(width, height) // 20))
            if self.debug:
                print(f"DEBUG: Using size {best_size} based on image dimensions")
            return best_size
            
        # Sort by variance (ascending)
        sorted_results = sorted(variance_results, key=lambda x: x[1])
        
        if self.debug:
            print(f"DEBUG: Top 5 lowest variance cell sizes:")
            for i, (size, variance) in enumerate(sorted_results[:5]):
                print(f"DEBUG:   #{i+1}: Size {size} - Variance: {variance:.2f}")
        
        # Look for significant drops in variance
        variance_drops = []
        for i in range(1, len(sorted_results)):
            prev_size, prev_var = sorted_results[i-1]
            curr_size, curr_var = sorted_results[i]
            
            # Calculate relative drop
            if prev_var > 0:
                rel_drop = (prev_var - curr_var) / prev_var
                variance_drops.append((curr_size, rel_drop))
        
        # Sort drops by magnitude (descending)
        if variance_drops:
            sorted_drops = sorted(variance_drops, key=lambda x: x[1], reverse=True)
            
            if self.debug:
                print(f"DEBUG: Top 3 variance drops:")
                for i, (size, drop) in enumerate(sorted_drops[:3]):
                    print(f"DEBUG:   #{i+1}: Size {size} - Relative drop: {drop:.3f}")
            
            # If there's a significant drop (>20%), use that size
            if sorted_drops[0][1] > 0.2:
                best_size = sorted_drops[0][0]
                if self.debug:
                    print(f"DEBUG: Selected size {best_size} based on significant variance drop")
                return best_size
        
        # Combine variance results with edge alignment scores
        combined_scores = []
        for size, var in sorted_results:
            # Find corresponding edge alignment score
            edge_score = next((score for s, score in edge_alignment_scores if s == size), 0)
            # Normalize variance (lower is better)
            norm_var = 1.0 - (var / sorted_results[-1][1]) if sorted_results[-1][1] > 0 else 0
            # Normalize edge score (higher is better)
            max_edge = max(score for _, score in edge_alignment_scores) if edge_alignment_scores else 1
            norm_edge = edge_score / max_edge if max_edge > 0 else 0
            # Combined score (weighted sum)
            combined = (0.6 * norm_var) + (0.4 * norm_edge)
            combined_scores.append((size, combined))
        
        # Sort by combined score (descending)
        sorted_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"DEBUG: Top 3 combined scores:")
            for i, (size, score) in enumerate(sorted_combined[:3]):
                print(f"DEBUG:   #{i+1}: Size {size} - Score: {score:.3f}")
        
        # Use the size with the best combined score
        best_size = sorted_combined[0][0]
        
        if self.debug:
            print(f"DEBUG: Selected best size from combined method: {best_size}")
                    
        return best_size
        
    def _find_best_offset(self, edges: np.ndarray, cell_size: int) -> Tuple[int, int]:
        """
        Find optimal grid offset for given cell size.
        
        Args:
            edges: Binary edge image
            cell_size: Detected cell size
            
        Returns:
            (x_offset, y_offset) tuple
        """
        height, width = edges.shape
        best_score = 0
        best_offset = (0, 0)
        
        # Limit search range to prevent freezing
        max_offset = min(32, cell_size)
        step_size = max(1, cell_size // 8)  # Sample fewer offsets for large cells
        
        for x_offset in range(0, max_offset, step_size):
            for y_offset in range(0, max_offset, step_size):
                # Score this offset based on edge alignment
                score = self._score_grid_alignment(edges, cell_size, (x_offset, y_offset))
                
                if score > best_score:
                    best_score = score
                    best_offset = (x_offset, y_offset)
                    
        return best_offset
        
    def _score_grid_alignment(self, edges: np.ndarray, cell_size: int, 
                            offset: Tuple[int, int]) -> float:
        """
        Score how well edges align with grid lines.
        
        Args:
            edges: Binary edge image
            cell_size: Cell size
            offset: Grid offset to test
            
        Returns:
            Alignment score (higher is better)
        """
        height, width = edges.shape
        x_offset, y_offset = offset
        
        total_score = 0
        line_count = 0
        
        # Score vertical grid lines
        for x in range(x_offset, width, cell_size):
            if x < width:
                line_strength = np.sum(edges[:, x])
                total_score += line_strength
                line_count += 1
                
        # Score horizontal grid lines  
        for y in range(y_offset, height, cell_size):
            if y < height:
                line_strength = np.sum(edges[y, :])
                total_score += line_strength
                line_count += 1
                
        return total_score / max(line_count, 1)
        
    def _crop_to_grid(self) -> np.ndarray:
        """
        Crop the image to the detected grid, preserving all pixel values.
        Returns:
            Cropped/Aligned image
        """
        height, width, channels = self.image.shape
        x_offset, y_offset = self.offset
        # Only keep the largest region that fits a whole number of cells
        x_max = ((width - x_offset) // self.cell_size) * self.cell_size + x_offset
        y_max = ((height - y_offset) // self.cell_size) * self.cell_size + y_offset
        cropped = self.image[y_offset:y_max, x_offset:x_max]
        return cropped
        
    def _find_modal_color(self, cell: np.ndarray) -> np.ndarray:
        """
        Find the most common color in a cell.
        
        Args:
            cell: Image cell as numpy array
            
        Returns:
            Modal color as RGB array
        """
        if cell.size == 0:
            return np.array([0, 0, 0])
            
        # Reshape to list of colors
        pixels = cell.reshape(-1, cell.shape[-1])
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Return most frequent color
        modal_idx = np.argmax(counts)
        return unique_colors[modal_idx]
        
    def _refine_artifacts(self) -> np.ndarray:
        """
        Refine reconstruction using neighbor majority vote.
        
        Returns:
            Final refined image
        """
        if self.grid_result is None:
            return self.image
            
        result = self.grid_result.copy()
        height, width = result.shape[:2]
        x_offset, y_offset = self.offset
        
        # Apply majority vote smoothing to reduce artifacts
        for y in range(y_offset, height, self.cell_size):
            for x in range(x_offset, width, self.cell_size):
                y_end = min(y + self.cell_size, height)
                x_end = min(x + self.cell_size, width)
                
                # Get neighboring cell colors
                neighbors = self._get_neighbor_colors(result, x, y, x_end, y_end)
                
                if len(neighbors) > 0:
                    # Find majority color among neighbors
                    majority_color = self._find_modal_color(np.array(neighbors))
                    
                    # Use neighbor majority if significantly different from current
                    current_color = result[y, x] if y < height and x < width else np.array([0, 0, 0])
                    
                    if not np.array_equal(current_color, majority_color):
                        # Only change if there's strong neighbor consensus
                        if len(neighbors) >= 3:
                            result[y:y_end, x:x_end] = majority_color
                            
        return result
        
    def _get_neighbor_colors(self, image: np.ndarray, x: int, y: int, 
                           x_end: int, y_end: int) -> list:
        """
        Get colors of neighboring cells.
        
        Args:
            image: Current image
            x, y: Cell top-left coordinates
            x_end, y_end: Cell bottom-right coordinates
            
        Returns:
            List of neighbor colors
        """
        height, width = image.shape[:2]
        neighbors = []
        
        # Check 8 neighboring cells
        for dy in [-self.cell_size, 0, self.cell_size]:
            for dx in [-self.cell_size, 0, self.cell_size]:
                if dx == 0 and dy == 0:
                    continue  # Skip self
                    
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < width and 0 <= ny < height):
                    neighbor_color = image[ny, nx]
                    neighbors.append(neighbor_color)
                    
        return neighbors


def create_validation_overlay(original_path: str, reconstructed: np.ndarray, 
                            cell_size: int, offset: Tuple[int, int]) -> np.ndarray:
    """
    Create side-by-side comparison with grid overlay for validation.
    
    Args:
        original_path: Path to original image
        reconstructed: Reconstructed image
        cell_size: Grid cell size
        offset: Grid offset
        
    Returns:
        Validation overlay image
    """
    # Load original
    with Image.open(original_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original = np.array(img)
    
    # Upscale reconstructed to original size for overlay
    h, w = original.shape[:2]
    rec = reconstructed
    rec_up = cv2.resize(rec, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create high-res difference overlay (red for differences)
    diff = np.any(original != rec_up, axis=-1)
    overlay = original.copy()
    overlay[diff] = [255, 0, 0]  # Red highlight for differences

    # Side-by-side: [original | overlay | upscaled reconstruction]
    comparison = np.hstack([original, overlay, rec_up])
    return comparison


def process_all_images(debug=False):
    """Process all images in the input directory."""
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No image files found in input/ directory")
        return
        
    print(f"Found {len(image_files)} image(s) to process")
    
    for image_file in image_files:
        try:
            # Process image
            reconstructor = PixelArtReconstructor(str(image_file), debug=debug)
            result = reconstructor.run()

            # Save reconstructed image
            output_path = output_dir / f"{image_file.stem}_reconstructed{image_file.suffix}"
            Image.fromarray(result).save(output_path)

            # Create and save validation overlay
            overlay = create_validation_overlay(
                str(image_file), result, 
                reconstructor.cell_size, reconstructor.offset
            )
            overlay_path = output_dir / f"{image_file.stem}_validation{image_file.suffix}"
            Image.fromarray(overlay).save(overlay_path)

            # --- Automated Output Analysis ---
            overlay_np = np.array(overlay)
            red_mask = (overlay_np[..., 0] == 255) & (overlay_np[..., 1] == 0) & (overlay_np[..., 2] == 0)
            percent_diff = 100.0 * np.sum(red_mask) / (overlay_np.shape[0] * overlay_np.shape[1])
            grid_size = reconstructor.cell_size
            img_h, img_w = overlay_np.shape[:2]
            grid_ratio = grid_size / min(img_h, img_w)
            warnings = []
            if percent_diff > 10.0:
                warnings.append(f"High difference: {percent_diff:.1f}% pixels differ from input.")
            if grid_ratio > 0.25:
                warnings.append(f"Grid size {grid_size} is large relative to image size {min(img_h, img_w)}.")
            if grid_size < 4:
                warnings.append(f"Grid size {grid_size} is very small (may be noise).")

            # Save debug grid detection plots
            reconstructor.debug_grid_detection(save_dir="output/grid_debug")

            print(f"Saved: {output_path} and {overlay_path}")
            if warnings:
                print("  [WARN] " + " ".join(warnings))
            else:
                print("  Output appears reasonable.")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    import sys
    debug_mode = "--debug" in sys.argv
    process_all_images(debug=debug_mode)
