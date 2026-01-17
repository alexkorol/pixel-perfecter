"""
Unified Pixel Art Reconstruction Pipeline

This module contains the consolidated, optimized pixel art reconstruction logic.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class PixelArtReconstructor:
    def debug_grid_detection(self, save_dir=None):
        """
        Save debug plots of edge projections and autocorrelations for this image.
        """
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            # Backend may already be initialised; fallback silently
            pass
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
    
    def __init__(self, image_path: Optional[str] = None, image: Optional[np.ndarray] = None, debug: bool = False):
        """
        Initialize with input image.
        
        Args:
            image_path: Path to input image file
            debug: Enable debug logging
        """
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided.")
        self.image_path = image_path
        self.image = image.copy() if image is not None else None
        self.cell_size = None
        self.offset = None
        self.grid_result = None
        self.debug = debug
        self._prefiltered_gray: Optional[np.ndarray] = None
        self._edge_map: Optional[np.ndarray] = None
        self._core_edge_map: Optional[np.ndarray] = None
        self.last_metrics: Optional[dict] = None
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        self.region_summaries: List[dict] = []
        self.mode: str = "global"

    def _report_progress(self, current: int, total: int, label: str) -> None:
        """Report refinement progress if a callback is registered."""
        if not self.progress_callback:
            return
        try:
            self.progress_callback(current, total, label)
        except Exception as exc:  # pylint: disable=broad-except
            if self.debug:
                print(f"DEBUG: Progress callback failed: {exc}")

    def _ensure_image_loaded(self) -> None:
        """Ensure self.image is populated."""
        if self.image is not None:
            return
        if self.image_path is None:
            raise ValueError("No image data available for reconstruction.")
        self.image = self._load_image()
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
    def run(self, mode: str = "global") -> np.ndarray:
        """
        Execute the reconstruction pipeline.
        Args:
            mode: "global" for single-grid reconstruction, "adaptive" for per-region grids.
        """
        self._ensure_image_loaded()
        mode = mode.lower()
        if mode not in {"global", "adaptive"}:
            raise ValueError(f"Unsupported reconstruction mode: {mode}")
        if mode == "adaptive":
            return self._run_adaptive_pipeline()
        return self._run_global_pipeline()

    # ------------------------- core pipelines ---------------------------

    def _run_global_pipeline(self) -> np.ndarray:
        print(f"Processing {self.image_path or '<in-memory image>'} [global]")
        print("Step 1: Detecting grid parameters (edge-projection peak analysis)...")
        self.cell_size, self.offset = self._detect_grid_parameters()
        if not self.cell_size or self.cell_size <= 0:
            self.cell_size = 4
        if self.offset is None:
            self.offset = (0, 0)
        print(f"  Detected cell size: {self.cell_size}, offset: {self.offset}")

        print("Step 2: Refining grid via reconstruction metrics...")
        refinement = self._refine_with_metric_search()
        self.cell_size = refinement["cell_size"]
        self.offset = refinement["offset"]
        reconstructed = refinement["reconstruction"]
        if reconstructed is None:
            reconstructed = self._empirical_pixel_reconstruction()
        self.last_metrics = refinement["metrics"]
        self.region_summaries = []
        self.mode = "global"
        if self.last_metrics is not None:
            self.last_metrics["mode"] = "global"
            self.last_metrics["regions"] = 1
        print(f"  Refined cell size: {self.cell_size}, offset: {self.offset}")
        print("Done!")
        return reconstructed

    def _run_adaptive_pipeline(self) -> np.ndarray:
        print(f"Processing {self.image_path or '<in-memory image>'} [adaptive]")
        regions = self._segment_regions()
        if not regions:
            print("  No distinct regions found; falling back to global reconstruction.")
            return self._run_global_pipeline()

        height, width = self.image.shape[:2]
        composite = np.zeros_like(self.image)
        coverage = np.zeros((height, width), dtype=bool)
        summaries: List[dict] = []

        total = len(regions)
        for idx, region in enumerate(regions, 1):
            x0, y0, x1, y1 = region["bbox"]
            region_mask = region["mask"].astype(bool)
            patch = self.image[y0:y1, x0:x1]

            self._report_progress(idx - 1, total, f"Region {idx}/{total}")

            try:
                patch_result, patch_mask, region_entries = self._reconstruct_region_recursive(
                    patch, region_mask, str(idx)
                )
            except Exception as exc:  # pylint: disable=broad-except
                if self.debug:
                    print(f"  [WARN] Region {idx} failed to reconstruct: {exc}")
                composite_patch = composite[y0:y1, x0:x1]
                composite_patch[region_mask] = patch[region_mask]
                composite[y0:y1, x0:x1] = composite_patch
                coverage_patch = coverage[y0:y1, x0:x1]
                coverage_patch[region_mask] = True
                coverage[y0:y1, x0:x1] = coverage_patch
                summaries.append(
                    {
                        "index": idx,
                        "status": "fallback",
                        "cell_size": None,
                        "offset": None,
                        "core_diff": None,
                        "area": int(np.count_nonzero(region_mask)),
                    }
                )
                continue

            composite_patch = composite[y0:y1, x0:x1]
            composite_patch[patch_mask] = patch_result[patch_mask]
            composite[y0:y1, x0:x1] = composite_patch
            coverage_patch = coverage[y0:y1, x0:x1]
            coverage_patch[patch_mask] = True
            coverage[y0:y1, x0:x1] = coverage_patch

            for entry in region_entries:
                enriched = dict(entry)
                enriched.setdefault("area", int(np.count_nonzero(region_mask)))
                summaries.append(enriched)

        # Fill uncovered pixels with original image to preserve background.
        composite[~coverage] = self.image[~coverage]

        # Update diagnostics
        self.cell_size = None
        self.offset = None
        self.region_summaries = summaries
        self.mode = "adaptive"

        try:
            self.last_metrics = self._evaluate_metrics_local(composite, 1, (0, 0))
        except Exception:
            self.last_metrics = None
        if self.last_metrics is not None:
            self.last_metrics["mode"] = "adaptive"
            self.last_metrics["regions"] = len(summaries)

        self._report_progress(total, total, "Refinement complete")
        print(f"  Processed {total} region(s).")
        print("Done!")
        return composite

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
            
    def _prepare_detection_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run halo-suppressing prefilter and edge detection steps."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=3)

        # Primary edge map keeps broader signals; core map suppresses halo bands.
        edges = cv2.Canny(filtered, 40, 120)
        cross_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        eroded = cv2.erode(edges, cross_kernel, iterations=1)
        strong_edges = cv2.Canny(filtered, 80, 200)
        core_edges = cv2.bitwise_or(eroded, strong_edges)
        return gray, filtered, edges, core_edges

    def _detect_grid_parameters(self) -> Tuple[int, Tuple[int, int]]:
        """
        Detect grid cell size and offset using edge projection analysis.
        
        Returns:
            Tuple of (cell_size, (x_offset, y_offset))
        """
        gray, filtered, edges, core_edges = self._prepare_detection_maps()
        self._prefiltered_gray = filtered
        self._edge_map = edges
        self._core_edge_map = core_edges
        
        # Find dominant cell size using projection analysis
        cell_size = self._find_dominant_period(core_edges)
        
        # Find best offset for this cell size
        offset = self._find_best_offset(core_edges, cell_size)
        
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
        prefiltered = (
            self._prefiltered_gray
            if self._prefiltered_gray is not None
            else cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=3)
        )
        
        # Test cell sizes from 4 to min(64, width//4, height//4)
        max_test_size = min(64, width//4, height//4)
        
        if self.debug:
            print(f"DEBUG: Variance-based detection testing sizes from 4 to {max_test_size}")
            
        variance_results = []
        edge_alignment_scores = []
        
        # Detect edges for alignment scoring
        base_edges = cv2.Canny(prefiltered, 40, 120)
        cross_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        eroded = cv2.erode(base_edges, cross_kernel, iterations=1)
        strong_edges = cv2.Canny(prefiltered, 80, 200)
        core_edges = cv2.bitwise_or(eroded, strong_edges)
        
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
                    score = self._score_grid_alignment(
                        core_edges, size, (offset_x, offset_y), halo_edges=base_edges
                    )
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
        
        halo_edges = self._edge_map if self._edge_map is not None else edges

        for x_offset in range(0, max_offset, step_size):
            for y_offset in range(0, max_offset, step_size):
                # Score this offset based on halo-suppressed alignment
                score = self._score_grid_alignment(
                    edges, cell_size, (x_offset, y_offset), halo_edges=halo_edges
                )
                
                if score > best_score:
                    best_score = score
                    best_offset = (x_offset, y_offset)
                    
        return best_offset
        
    def _score_grid_alignment(
        self,
        edges: np.ndarray,
        cell_size: int,
        offset: Tuple[int, int],
        *,
        halo_edges: Optional[np.ndarray] = None,
    ) -> float:
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
        halo_map = halo_edges if halo_edges is not None else edges
        
        # Score vertical grid lines
        for x in range(x_offset, width, cell_size):
            if x < width:
                line_strength = np.sum(edges[:, x])
                halo_penalty = 0
                if x - 1 >= 0:
                    halo_penalty += 0.5 * np.sum(halo_map[:, x - 1])
                if x + 1 < width:
                    halo_penalty += 0.5 * np.sum(halo_map[:, x + 1])
                line_strength = max(line_strength - halo_penalty, 0)
                total_score += line_strength
                line_count += 1
                
        # Score horizontal grid lines  
        for y in range(y_offset, height, cell_size):
            if y < height:
                line_strength = np.sum(edges[y, :])
                halo_penalty = 0
                if y - 1 >= 0:
                    halo_penalty += 0.5 * np.sum(halo_map[y - 1, :])
                if y + 1 < height:
                    halo_penalty += 0.5 * np.sum(halo_map[y + 1, :])
                line_strength = max(line_strength - halo_penalty, 0)
                total_score += line_strength
                line_count += 1
                
        return total_score / max(line_count, 1)

    def _generate_size_candidates(self, base_size: int) -> List[int]:
        """Create a set of plausible cell sizes near the detected period."""
        if self.image is None:
            return [base_size]

        height, width = self.image.shape[:2]
        max_candidate = max(4, min(192, min(height, width)))
        candidates = set()
        candidates.add(max(4, base_size))

        # Explore nearby sizes (+/- 12px) to capture harmonics/off-by-one cases.
        for delta in range(-12, 13):
            candidate = base_size + delta
            if 4 <= candidate <= max_candidate:
                candidates.add(candidate)

        # Include divisors and multiples to handle harmonic mis-detections.
        for divisor in (2, 3, 4):
            if base_size % divisor == 0:
                candidate = base_size // divisor
                if 4 <= candidate <= max_candidate:
                    candidates.add(candidate)

        for multiplier in (2, 3):
            candidate = base_size * multiplier
            if 4 <= candidate <= max_candidate:
                candidates.add(candidate)

        ordered = sorted(candidates, key=lambda s: (abs(s - base_size), s))
        return ordered[: min(len(ordered), 15)]

    def _collect_offset_candidates(
        self,
        cell_size: int,
        initial_offset: Tuple[int, int],
        extra_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Return a bounded list of offset candidates near the initial estimate."""
        radius = max(1, min(6, cell_size // 6))
        offsets: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()

        def _add(offset: Tuple[int, int]) -> None:
            ox = int(round(offset[0]))
            oy = int(round(offset[1]))
            if ox < 0 or oy < 0 or ox >= cell_size or oy >= cell_size:
                return
            key = (ox, oy)
            if key in seen:
                return
            seen.add(key)
            offsets.append(key)

        _add(initial_offset)
        if extra_offsets:
            for extra in extra_offsets:
                _add(extra)

        base_x, base_y = initial_offset
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                _add((base_x + dx, base_y + dy))

        max_offsets = min(25, len(offsets))
        return offsets[:max_offsets]

    def _segment_regions(self) -> List[dict]:
        """Segment the image into candidate regions for local lattice fitting."""
        self._ensure_image_loaded()
        if self._edge_map is None or self._core_edge_map is None:
            _, _, self._edge_map, self._core_edge_map = self._prepare_detection_maps()

        edges = self._core_edge_map.copy()
        if edges is None:
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)

        num_labels, labels = cv2.connectedComponents(thresh)
        height, width = edges.shape
        regions: List[dict] = []

        for label in range(1, num_labels):
            mask = labels == label
            area = int(np.sum(mask))
            if area < 64:
                continue
            ys, xs = np.where(mask)
            y0 = max(int(ys.min()) - 2, 0)
            y1 = min(int(ys.max()) + 3, height)
            x0 = max(int(xs.min()) - 2, 0)
            x1 = min(int(xs.max()) + 3, width)
            sub_mask = mask[y0:y1, x0:x1]
            split_masks = self._split_region_watershed(sub_mask)
            if len(split_masks) <= 1:
                regions.append({"bbox": (x0, y0, x1, y1), "mask": sub_mask})
                continue

            for split in split_masks:
                if np.count_nonzero(split) < 64:
                    continue
                ys2, xs2 = np.where(split)
                sy0 = y0 + int(ys2.min())
                sy1 = y0 + int(ys2.max()) + 1
                sx0 = x0 + int(xs2.min())
                sx1 = x0 + int(xs2.max()) + 1
                cropped_mask = split[int(ys2.min()):int(ys2.max()) + 1, int(xs2.min()):int(xs2.max()) + 1]
                regions.append(
                    {
                        "bbox": (sx0, sy0, sx1, sy1),
                        "mask": cropped_mask,
                    }
                )

        regions.sort(key=lambda seg: -np.sum(seg["mask"]))
        return regions

    def _split_region_watershed(self, mask: np.ndarray) -> List[np.ndarray]:
        """Split a binary mask using watershed to separate touching sprites."""
        mask_uint8 = (mask.astype(np.uint8)) * 255
        if np.count_nonzero(mask_uint8) < 128:
            return [mask]

        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        max_val = dist.max()
        if max_val < 4.0:
            return [mask]

        _, sure_fg = cv2.threshold(dist, 0.45 * max_val, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        if np.count_nonzero(sure_fg) == 0:
            return [mask]

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask_uint8, kernel, iterations=1)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        if num_markers <= 1:
            return [mask]

        markers = markers + 1
        markers[unknown == 255] = 0
        mask_color = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
        cv2.watershed(mask_color, markers)

        splits: List[np.ndarray] = []
        for label in range(2, markers.max() + 1):
            sub = markers == label
            if np.count_nonzero(sub) < 64:
                continue
            splits.append(sub.astype(bool))

        return splits or [mask]

    @staticmethod
    def _split_mask_quadrants(mask: np.ndarray) -> List[np.ndarray]:
        """Fallback split: divide the mask into quadrants."""
        h, w = mask.shape
        mid_y = h // 2
        mid_x = w // 2
        slices = (
            (slice(0, mid_y), slice(0, mid_x)),
            (slice(0, mid_y), slice(mid_x, w)),
            (slice(mid_y, h), slice(0, mid_x)),
            (slice(mid_y, h), slice(mid_x, w)),
        )
        quads: List[np.ndarray] = []
        for y_slice, x_slice in slices:
            sub_region = mask[y_slice, x_slice]
            if np.count_nonzero(sub_region) < 32:
                continue
            sub_mask = np.zeros_like(mask, dtype=bool)
            sub_mask[y_slice, x_slice] = sub_region
            quads.append(sub_mask)
        return quads or [mask]

    def _reconstruct_region_recursive(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
        region_id: str,
        depth: int = 0,
        max_depth: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Recursively reconstruct a region, subdividing when residuals are high."""
        mask_bool = mask.astype(bool)
        coverage = mask_bool.copy()
        area = int(np.count_nonzero(mask_bool))
        area_threshold = 48
        h, w = patch.shape[:2]

        if area < area_threshold or min(h, w) < 8:
            return patch.copy(), coverage, [
                {
                    "index": region_id,
                    "status": "small",
                    "cell_size": None,
                    "offset": None,
                    "core_diff": None,
                    "percent_diff": 0.0,
                    "area": area,
                }
            ]

        try:
            fitted_patch, metrics = self._fit_region_quick(patch, mask_bool, region_id)
        except Exception as exc:  # pylint: disable=broad-except
            if self.debug:
                print(f"  [WARN] Quick fit failed for region {region_id}: {exc}")
            return patch.copy(), coverage, [
                {
                    "index": region_id,
                    "status": "fallback",
                    "cell_size": None,
                    "offset": None,
                    "core_diff": None,
                    "percent_diff": 0.0,
                    "area": area,
                }
            ]

        percent_diff = float(metrics.get("percent_diff", 100.0))
        result_patch = patch.copy()
        result_patch[mask_bool] = fitted_patch[mask_bool]

        summary_entry = {
            "index": region_id,
            "status": metrics.get("status", "ok"),
            "cell_size": metrics.get("cell_size"),
            "offset": metrics.get("offset"),
            "core_diff": percent_diff,
            "percent_diff": percent_diff,
            "mean_error": metrics.get("mean_error", 0.0),
            "area": area,
        }

        should_split = (
            percent_diff > 5.0
            and depth < max_depth
            and min(h, w) >= 14
            and area > area_threshold * 2
        )

        if not should_split:
            return result_patch, coverage, [summary_entry]

        splits = self._split_region_watershed(mask_bool)
        if len(splits) <= 1:
            splits = self._split_mask_quadrants(mask_bool)

        if len(splits) <= 1:
            summary_entry["status"] = "unsplit"
            return result_patch, coverage, [summary_entry]

        combined_patch = patch.copy()
        combined_mask = np.zeros_like(mask_bool)
        summaries: List[dict] = []
        child_idx = 1

        for sub_mask in splits:
            if np.count_nonzero(sub_mask) < area_threshold:
                continue
            ys, xs = np.where(sub_mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            sub_patch = patch[y0:y1, x0:x1]
            sub_mask_crop = sub_mask[y0:y1, x0:x1]

            child_id = f"{region_id}.{child_idx}"
            child_idx += 1

            child_patch, child_coverage, child_summaries = self._reconstruct_region_recursive(
                sub_patch, sub_mask_crop, child_id, depth + 1, max_depth
            )

            target_region = combined_patch[y0:y1, x0:x1]
            target_region[child_coverage] = child_patch[child_coverage]
            combined_patch[y0:y1, x0:x1] = target_region

            combined_mask[y0:y1, x0:x1][child_coverage] = True
            summaries.extend(child_summaries)

        leftover = np.logical_and(mask_bool, np.logical_not(combined_mask))
        if np.any(leftover):
            combined_patch[leftover] = result_patch[leftover]
            combined_mask[leftover] = True

        summary_entry["status"] = "split"
        return combined_patch, combined_mask, [summary_entry] + summaries

    def _refine_with_metric_search(self) -> Dict[str, object]:
        """Refine detected parameters using reconstruction difference metrics."""
        if self.cell_size is None or self.cell_size <= 0:
            self.cell_size = 4
        if self.offset is None:
            self.offset = (0, 0)
        if self.image is None:
            raise ValueError("Image must be loaded before refinement.")

        base_size = int(self.cell_size or 4)
        base_offset = self.offset

        # Ensure edge maps are available for offset search.
        if self._core_edge_map is None or self._edge_map is None:
            _, _, self._edge_map, self._core_edge_map = self._prepare_detection_maps()

        size_candidates = self._generate_size_candidates(base_size)
        size_candidates = self._divisor_cascade(size_candidates, base_size)

        evaluations: List[Tuple[int, float, List[Tuple[int, int]]]] = []
        for size in size_candidates:
            offset_guess = self._find_best_offset(self._core_edge_map, size)
            extras: Optional[List[Tuple[int, int]]] = None
            if size == base_size:
                extras = [base_offset]
            offsets = self._collect_offset_candidates(size, offset_guess, extras)
            alignment_score = self._score_grid_alignment(
                self._core_edge_map, size, offset_guess, halo_edges=self._edge_map
            )
            evaluations.append((size, alignment_score, offsets))

        if not evaluations:
            self._report_progress(1, 1, "Refinement skipped")
            reconstruction = self._empirical_pixel_reconstruction()
            metrics = self._evaluate_metrics_local(reconstruction, base_size, base_offset)
            return {
                "cell_size": base_size,
                "offset": base_offset,
                "reconstruction": reconstruction,
                "metrics": metrics,
            }

        evaluations.sort(key=lambda item: item[1], reverse=True)
        MAX_SIZE_EVALS = 10
        top_evals = evaluations[:MAX_SIZE_EVALS]
        if not any(size == base_size for size, _, _ in top_evals):
            base_entry = next(((size, score, offsets) for size, score, offsets in evaluations if size == base_size), None)
            if base_entry:
                top_evals = [base_entry] + [item for item in top_evals if item[0] != base_size]
                top_evals = top_evals[:MAX_SIZE_EVALS]

        total_offsets = sum(len(offsets) for _, _, offsets in top_evals)
        if total_offsets == 0:
            self._report_progress(1, 1, "Refinement skipped")
            reconstruction = self._empirical_pixel_reconstruction()
            metrics = self._evaluate_metrics_local(reconstruction, base_size, base_offset)
            return {
                "cell_size": base_size,
                "offset": base_offset,
                "reconstruction": reconstruction,
                "metrics": metrics,
            }

        best_metrics: Optional[dict] = None
        best_reconstruction: Optional[np.ndarray] = None
        best_size = base_size
        best_offset = base_offset

        completed = 0
        EARLY_EXIT_THRESHOLD = 0.5
        early_exit = False

        for size, _, offsets in top_evals:
            for offset in offsets:
                completed += 1
                self._report_progress(
                    completed,
                    total_offsets,
                    f"{size}px @ ({offset[0]}, {offset[1]})",
                )
                reconstruction, metrics = self._reconstruct_and_score(size, offset)
                if reconstruction is None or metrics is None:
                    continue
                if best_metrics is None or self._is_metrics_better(metrics, best_metrics):
                    best_metrics = metrics
                    best_reconstruction = reconstruction
                    best_size = size
                    best_offset = offset
                    core_diff = metrics.get("percent_diff_core", 100.0)
                    if core_diff < EARLY_EXIT_THRESHOLD:
                        early_exit = True
                        break
                else:
                    continue
            if early_exit:
                break

        self._report_progress(total_offsets, total_offsets, "Refinement complete")

        if best_metrics is None or best_reconstruction is None:
            # Fallback to the originally detected parameters.
            reconstruction = self._empirical_pixel_reconstruction()
            metrics = self._evaluate_metrics_local(reconstruction, base_size, base_offset)
            return {
                "cell_size": base_size,
                "offset": base_offset,
                "reconstruction": reconstruction,
                "metrics": metrics,
            }

        return {
            "cell_size": best_size,
            "offset": best_offset,
            "reconstruction": best_reconstruction,
            "metrics": best_metrics,
        }

    def _reconstruct_and_score(
        self, cell_size: int, offset: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """Reconstruct an image with explicit parameters and compute diff metrics."""
        prev_size, prev_offset = self.cell_size, self.offset
        try:
            self.cell_size = cell_size
            self.offset = offset
            reconstruction = self._empirical_pixel_reconstruction()
            metrics = self._evaluate_metrics_local(reconstruction, cell_size, offset)
            return reconstruction, metrics
        except Exception as exc:  # pylint: disable=broad-except
            if self.debug:
                print(f"DEBUG: Reconstruction failed for size {cell_size}, offset {offset}: {exc}")
            return None, None
        finally:
            self.cell_size = prev_size
            self.offset = prev_offset

    def _evaluate_metrics_local(
        self, reconstruction: np.ndarray, cell_size: int, offset: Tuple[int, int]
    ) -> dict:
        """Compute difference metrics without reloading images from disk."""
        if self.image is None:
            raise ValueError("Image must be loaded before evaluating metrics.")

        original = self.image
        h, w = original.shape[:2]
        rec_up = cv2.resize(reconstruction, (w, h), interpolation=cv2.INTER_NEAREST)
        diff_ctx = _build_diff_context(original, rec_up, cell_size, offset)
        metrics = _compute_overlay_metrics(diff_ctx, cell_size, offset)
        return metrics

    @staticmethod
    def _is_metrics_better(candidate: Optional[dict], current: Optional[dict]) -> bool:
        """Determine if candidate metrics are preferable."""
        if candidate is None:
            return False
        if current is None:
            return True

        candidate_core = candidate.get("percent_diff_core", float("inf"))
        current_core = current.get("percent_diff_core", float("inf"))

        if candidate_core < current_core - 0.5:
            return True
        if candidate_core > current_core + 0.5:
            return False

        candidate_total = candidate.get("percent_diff_total", candidate.get("percent_diff", float("inf")))
        current_total = current.get("percent_diff_total", current.get("percent_diff", float("inf")))
        return candidate_total < current_total

    # ------------------------------ heuristics ----------------------------

    def _estimate_cell_variance(self, reconstruction: np.ndarray) -> float:
        """Estimate intra-cell variance against the original image."""
        if self.image is None:
            return float("inf")
        original = self.image
        h, w = original.shape[:2]
        rec_up = cv2.resize(reconstruction, (w, h), interpolation=cv2.INTER_NEAREST)
        diff = np.abs(original.astype(np.int32) - rec_up.astype(np.int32))
        return float(np.mean(diff))

    def _divisor_cascade(self, candidates: Iterable[int], base_size: int) -> List[int]:
        """Augment size candidates with divisor cascade when core variance is high."""
        if self.image is None:
            return list(candidates)

        ordered = list(dict.fromkeys(candidates))
        try:
            idx = ordered.index(base_size)
        except ValueError:
            idx = 0

        test_sizes = ordered[: min(len(ordered), 5)]

        suspicious_sizes: List[int] = []
        for size in test_sizes:
            if size <= 4:
                continue
            # Compute a quick reconstruction for variance check.
            prev_size, prev_offset = self.cell_size, self.offset
            try:
                self.cell_size = size
                self.offset = self._find_best_offset(
                    self._core_edge_map if self._core_edge_map is not None else self._edge_map,
                    size,
                )
                recon = self._empirical_pixel_reconstruction()
                variance = self._estimate_cell_variance(recon)
            except Exception:
                variance = float("inf")
            finally:
                self.cell_size = prev_size
                self.offset = prev_offset

            if variance > 25.0:  # High variance indicates unresolved substructure.
                suspicious_sizes.append(size)

        cascade_sizes: List[int] = []
        for size in suspicious_sizes:
            div = size
            while div >= 4:
                div //= 2
                if div >= 4 and div not in ordered and div not in cascade_sizes:
                    cascade_sizes.append(div)

        return ordered + cascade_sizes

    @staticmethod
    def _evaluate_region_diff(
        original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute difference statistics limited to the masked region."""
        diff_map = np.any(original != reconstructed, axis=-1)
        masked_area = int(np.count_nonzero(mask))
        diff_count = int(np.count_nonzero(np.logical_and(diff_map, mask)))
        percent_diff = 100.0 * diff_count / max(masked_area, 1)

        error = np.abs(
            original.astype(np.int32) - reconstructed.astype(np.int32)
        )
        if error.ndim == 3:
            error = np.mean(error, axis=-1)
        mean_error = float(np.mean(error[mask])) if masked_area > 0 else 0.0

        return {
            "percent_diff": percent_diff,
            "mean_error": mean_error,
        }

    def _fit_region_quick(
        self, patch: np.ndarray, mask: np.ndarray, region_id: str
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Fit a lattice to a region using lightweight candidate search."""
        local = PixelArtReconstructor(image=patch, debug=self.debug)
        local.progress_callback = None

        try:
            local.cell_size, local.offset = local._detect_grid_parameters()
        except Exception:  # pylint: disable=broad-except
            return patch.copy(), {
                "status": "raw",
                "cell_size": None,
                "offset": None,
                "percent_diff": 0.0,
            }

        base_size = int(local.cell_size or 4)
        height, width = patch.shape[:2]
        max_candidate = max(4, min(height, width))

        candidate_sizes = [base_size]
        if base_size >= 8:
            candidate_sizes.append(base_size // 2)
        if base_size >= 12:
            candidate_sizes.append(base_size // 3)
        if base_size * 2 <= max_candidate:
            candidate_sizes.append(base_size * 2)
        if base_size * 3 <= max_candidate:
            candidate_sizes.append(base_size * 3)
        candidate_sizes.append(4)

        unique_sizes = []
        for size in candidate_sizes:
            size_int = int(size)
            if size_int >= 4 and size_int <= max_candidate and size_int not in unique_sizes:
                unique_sizes.append(size_int)

        best_patch = patch.copy()
        best_metrics: Optional[Dict[str, object]] = None
        local._ensure_image_loaded()
        _, _, local_edges, local_core_edges = local._prepare_detection_maps()
        local._edge_map = local_edges
        local._core_edge_map = local_core_edges

        for size in unique_sizes:
            offset = local._find_best_offset(local._core_edge_map, size)
            local.cell_size = size
            local.offset = offset
            try:
                recon = local._empirical_pixel_reconstruction()
            except Exception:  # pylint: disable=broad-except
                continue

            upscaled = cv2.resize(
                recon, (width, height), interpolation=cv2.INTER_NEAREST
            )
            metrics = self._evaluate_region_diff(patch, upscaled, mask)
            metrics.update(
                {
                    "cell_size": size,
                    "offset": offset,
                }
            )

            if (
                best_metrics is None
                or metrics["percent_diff"] < best_metrics["percent_diff"]
                or (
                    metrics["percent_diff"] == best_metrics["percent_diff"]
                    and metrics["mean_error"] < best_metrics["mean_error"]
                )
            ):
                best_metrics = metrics
                best_patch = upscaled

                if metrics["percent_diff"] <= 1.0:
                    break

        if best_metrics is None:
            best_metrics = {
                "status": "raw",
                "cell_size": None,
                "offset": None,
                "percent_diff": 0.0,
            }

        best_metrics.setdefault("status", "ok")
        best_metrics["region_id"] = region_id
        return best_patch, best_metrics
        
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
        
    def _extract_core_pixels(self, cell: np.ndarray) -> np.ndarray:
        """Return pixels from the interior of a cell, trimming halo bands."""
        h, w = cell.shape[:2]
        if h == 0 or w == 0:
            return np.empty((0, 3), dtype=np.uint8)

        if self.cell_size:
            # Keep at least one pixel margin but cap so we preserve small cells.
            margin = max(1, min(self.cell_size // 6, (min(h, w) - 1) // 2))
        else:
            margin = 1

        if margin <= 0 or h <= 2 * margin or w <= 2 * margin:
            return cell.reshape(-1, cell.shape[-1])

        core = cell[margin : h - margin, margin : w - margin]
        if core.size == 0:
            return cell.reshape(-1, cell.shape[-1])
        return core.reshape(-1, core.shape[-1])

    def _find_modal_color(self, cell: np.ndarray) -> np.ndarray:
        """
        Find a consensus color within a cell using halo-trimmed vector median.
        """
        if cell.size == 0:
            return np.array([0, 0, 0], dtype=np.uint8)

        core_pixels = self._extract_core_pixels(cell)
        if core_pixels.size == 0:
            core_pixels = cell.reshape(-1, cell.shape[-1])

        # Use per-channel median and pick the closest observed colour (fast L1 approximation).
        median = np.median(core_pixels, axis=0)
        deltas = np.abs(core_pixels.astype(np.int32) - median.astype(np.int32))
        distances = np.sum(deltas, axis=1)
        best_idx = int(np.argmin(distances))
        consensus = core_pixels[best_idx]

        return consensus.astype(np.uint8)
        
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


def _build_diff_context(
    original: np.ndarray, rec_up: np.ndarray, cell_size: int, offset: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """Create masks that separate halo and core regions for diagnostics."""
    diff = np.any(original != rec_up, axis=-1)
    h, w = diff.shape
    grid_mask = np.zeros_like(diff, dtype=bool)
    core_mask = np.zeros_like(diff, dtype=bool)

    cell_size = max(int(cell_size), 1)
    x_offset, y_offset = offset

    for y in range(y_offset, h, cell_size):
        y_end = min(y + cell_size, h)
        for x in range(x_offset, w, cell_size):
            x_end = min(x + cell_size, w)
            grid_mask[y:y_end, x:x_end] = True

            cell_h = y_end - y
            cell_w = x_end - x
            if cell_h <= 2 or cell_w <= 2:
                continue

            margin = max(1, min(cell_size // 6, (min(cell_h, cell_w) - 1) // 2))
            if margin <= 0:
                continue

            inner_y0 = y + margin
            inner_y1 = y_end - margin
            inner_x0 = x + margin
            inner_x1 = x_end - margin
            if inner_y1 > inner_y0 and inner_x1 > inner_x0:
                core_mask[inner_y0:inner_y1, inner_x0:inner_x1] = True

    core_mask &= grid_mask
    halo_mask = np.logical_and(grid_mask, np.logical_not(core_mask))

    core_diff = np.logical_and(diff, core_mask)
    halo_diff = np.logical_and(diff, halo_mask)
    outside_diff = np.logical_and(diff, np.logical_not(grid_mask))

    return {
        "diff": diff,
        "grid_mask": grid_mask,
        "core_mask": core_mask,
        "halo_mask": halo_mask,
        "core_diff": core_diff,
        "halo_diff": halo_diff,
        "outside_diff": outside_diff,
    }


def _assemble_overlay(original: np.ndarray, overlay_view: np.ndarray, rec_up: np.ndarray) -> np.ndarray:
    """Stack original, diagnostic overlay, and reconstruction panels."""
    return np.hstack([original, overlay_view, rec_up])


def _render_diagnostic_overlays(
    original: np.ndarray, rec_up: np.ndarray, diff_ctx: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Generate multiple overlay variants for inspection."""
    overlays: Dict[str, np.ndarray] = {}

    raw_overlay = original.copy()
    raw_overlay[diff_ctx["diff"]] = [255, 0, 0]
    overlays["raw"] = _assemble_overlay(original, raw_overlay, rec_up)

    combined_overlay = original.copy()
    combined_overlay[diff_ctx["halo_diff"]] = [255, 255, 0]
    combined_overlay[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["combined"] = _assemble_overlay(original, combined_overlay, rec_up)

    halo_suppressed = original.copy()
    halo_suppressed[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["halo_suppressed"] = _assemble_overlay(original, halo_suppressed, rec_up)

    core_only = original.astype(np.float32)
    core_mask = diff_ctx["core_mask"]
    core_only[~core_mask] *= 0.25
    core_only = np.clip(core_only, 0, 255).astype(np.uint8)
    core_only[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["core_only"] = _assemble_overlay(original, core_only, rec_up)

    return overlays


def _compute_overlay_metrics(
    diff_ctx: Dict[str, np.ndarray], cell_size: int, offset: Tuple[int, int]
) -> dict:
    """Compute difference statistics using halo/core separation."""
    total_pixels = diff_ctx["diff"].size
    total_diff = int(np.sum(diff_ctx["diff"]))
    core_pixels = int(np.sum(diff_ctx["core_mask"]))
    halo_pixels = int(np.sum(diff_ctx["halo_mask"]))
    outside_pixels = total_pixels - int(np.sum(diff_ctx["grid_mask"]))

    core_diff = int(np.sum(diff_ctx["core_diff"]))
    halo_diff = int(np.sum(diff_ctx["halo_diff"]))
    outside_diff = int(np.sum(diff_ctx["outside_diff"]))

    percent_total = 100.0 * total_diff / max(total_pixels, 1)
    percent_core = 100.0 * core_diff / max(core_pixels, 1) if core_pixels else 0.0
    percent_halo = 100.0 * halo_diff / max(halo_pixels, 1) if halo_pixels else 0.0
    percent_outside = 100.0 * outside_diff / max(outside_pixels, 1) if outside_pixels else 0.0

    h, w = diff_ctx["diff"].shape
    grid_ratio = cell_size / max(1, min(h, w))

    warnings: List[str] = []
    if percent_core > 5.0:
        warnings.append(f"Core mismatch {percent_core:.1f}% - inspect alignment.")
    if percent_halo > 15.0 and percent_core < 3.0:
        warnings.append("Differences dominated by halo glow.")
    if percent_outside > 2.0:
        warnings.append("Differences detected outside projected grid.")
    if grid_ratio > 0.25:
        warnings.append(
            f"Grid size {cell_size} is large relative to image size {min(h, w)}."
        )
    if cell_size < 4:
        warnings.append("Grid size is very small (may indicate noise).")

    return {
        "cell_size": cell_size,
        "offset": offset,
        "percent_diff_total": float(percent_total),
        "percent_diff_core": float(percent_core),
        "percent_diff_halo": float(percent_halo),
        "percent_diff_outside": float(percent_outside),
        "percent_diff": float(percent_total),  # Backwards compatibility
        "grid_ratio": float(grid_ratio),
        "warnings": warnings,
    }


def build_validation_diagnostics(
    original_path: str,
    reconstructed: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
) -> Dict[str, object]:
    """Produce overlays and metrics for validation/GUI tooling."""
    with Image.open(original_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        original = np.array(img)

    h, w = original.shape[:2]
    rec_up = cv2.resize(reconstructed, (w, h), interpolation=cv2.INTER_NEAREST)
    diff_ctx = _build_diff_context(original, rec_up, cell_size, offset)
    overlays = _render_diagnostic_overlays(original, rec_up, diff_ctx)
    metrics = _compute_overlay_metrics(diff_ctx, cell_size, offset)

    return {
        "original": original,
        "upscaled_reconstruction": rec_up,
        "overlays": overlays,
        "diff_context": diff_ctx,
        "metrics": metrics,
    }


def create_validation_overlay(
    original_path: str,
    reconstructed: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    mode: str = "combined",
) -> np.ndarray:
    """Return a specific diagnostic overlay variant."""
    diagnostics = build_validation_diagnostics(original_path, reconstructed, cell_size, offset)
    overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
    if mode not in overlays:
        mode = "combined"
    return overlays[mode]


def _evaluate_with_parameters(
    image_path: Path,
    cell_size: int,
    offset: Tuple[int, int],
    debug: bool,
) -> dict:
    """Evaluate reconstruction quality for explicit parameters."""
    candidate = PixelArtReconstructor(str(image_path), debug=debug)
    candidate.image = candidate._load_image()
    if candidate.image is None:
        raise ValueError(f"Could not load image: {image_path}")

    candidate.cell_size = cell_size
    candidate.offset = offset
    result = candidate._empirical_pixel_reconstruction()
    diagnostics = build_validation_diagnostics(str(image_path), result, cell_size, offset)
    overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
    metrics = diagnostics["metrics"]  # type: ignore[assignment]
    overlay = overlays["combined"]
    return {
        "result": result,
        "overlay": overlay,
        "overlays": overlays,
        "metrics": metrics,
        "reconstructor": candidate,
    }


def process_all_images(debug: bool = False, use_ml: bool = False, adaptive: bool = False):
    """Process all images in the input directory."""
    input_dir = Path("input")
    output_dir = Path("output")
    grid_debug_dir = output_dir / "grid_debug"

    output_dir.mkdir(exist_ok=True)
    grid_debug_dir.mkdir(exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("No image files found in input/ directory")
        return

    suggest_fn = None
    if use_ml:
        try:
            from ml.inference import suggest_parameters as _suggest_parameters  # type: ignore

            suggest_fn = _suggest_parameters
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] ML suggestions unavailable: {exc}")
            suggest_fn = None

    print(f"Found {len(image_files)} image(s) to process")
    metrics_records: List[dict] = []

    for image_file in image_files:
        try:
            reconstructor = PixelArtReconstructor(str(image_file), debug=debug)
            mode = "adaptive" if adaptive else "global"
            result = reconstructor.run(mode=mode)

            cell_size_for_metrics = reconstructor.cell_size if reconstructor.cell_size else 1
            offset_for_metrics = reconstructor.offset if reconstructor.offset else (0, 0)
            diagnostics = build_validation_diagnostics(
                str(image_file), result, cell_size_for_metrics, offset_for_metrics
            )
            overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
            metrics = diagnostics["metrics"]  # type: ignore[assignment]

            best = {
                "result": result,
                "overlay": overlays["combined"],
                "overlays": overlays,
                "metrics": metrics,
                "reconstructor": reconstructor,
            }
            ml_applied = False

            if suggest_fn and not adaptive:
                suggestions = suggest_fn(str(image_file))
                for suggestion in suggestions:
                    candidate = _evaluate_with_parameters(
                        image_path=image_file,
                        cell_size=suggestion.cell_size,
                        offset=suggestion.offset,
                        debug=debug,
                    )
                    if (
                        candidate["metrics"]["percent_diff"] + 0.1
                        < best["metrics"]["percent_diff"]
                    ):
                        candidate["metrics"]["warnings"].append(
                            f"Applied ML suggestion (conf={suggestion.confidence:.2f})"
                        )
                        best = candidate
                        ml_applied = True

            result = best["result"]
            overlay = best["overlay"]
            metrics = best["metrics"]
            reconstructor = best["reconstructor"]

            output_path = output_dir / f"{image_file.stem}_reconstructed{image_file.suffix}"
            Image.fromarray(result).save(output_path)

            overlay_path = output_dir / f"{image_file.stem}_validation{image_file.suffix}"
            Image.fromarray(overlay).save(overlay_path)

            reconstructor.debug_grid_detection(save_dir=str(grid_debug_dir))

            print(f"Saved: {output_path} and {overlay_path}")
            if metrics["warnings"]:
                print("  [WARN] " + " ".join(metrics["warnings"]))
            else:
                print("  Output appears reasonable.")
            if ml_applied:
                print("  [INFO] ML suggestion applied.")

            metrics_records.append(
                {
                    "image": image_file.name,
                    "cell_size": metrics.get("cell_size"),
                    "offset_x": metrics.get("offset", (0, 0))[0],
                    "offset_y": metrics.get("offset", (0, 0))[1],
                    "percent_diff": round(metrics.get("percent_diff", 0.0), 4),
                    "percent_diff_core": round(metrics.get("percent_diff_core", 0.0), 4),
                    "percent_diff_halo": round(metrics.get("percent_diff_halo", 0.0), 4),
                    "percent_diff_outside": round(metrics.get("percent_diff_outside", 0.0), 4),
                    "grid_ratio": round(metrics.get("grid_ratio", 0.0), 4),
                    "warnings": " | ".join(metrics.get("warnings", []))
                    if metrics.get("warnings")
                    else "",
                    "ml_used": "yes" if ml_applied else ("n/a" if adaptive else "no"),
                    "mode": reconstructor.mode,
                    "regions": len(reconstructor.region_summaries)
                    if reconstructor.region_summaries
                    else 1,
                }
            )

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing {image_file}: {e}")

    if metrics_records:
        metrics_path = output_dir / "metrics.csv"
        with metrics_path.open("w", newline="") as csvfile:
            fieldnames = [
                "image",
                "cell_size",
                "offset_x",
                "offset_y",
                "percent_diff",
                "percent_diff_core",
                "percent_diff_halo",
                "percent_diff_outside",
                "grid_ratio",
                "warnings",
                "mode",
                "regions",
                "ml_used",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_records)
        print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all images with pixel reconstructor.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Use ML suggestions to refine reconstructions when available.",
    )
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive per-region grids.")
    args = parser.parse_args()
    process_all_images(debug=args.debug, use_ml=args.ml, adaptive=args.adaptive)
