# Testing Methodology for Grid Detection and Pixel Art Reconstruction

This document outlines the key testing strategies developed during the pixel-perfecter research to validate grid detection accuracy and artifact removal.

## 1. Tests for Grid Size & Offset Accuracy

These tests were crucial for solving the primary problem: correctly identifying the underlying grid of the "implied pixels."

| Test Name | How it Works (The "Test") | What it's Good For |
| :--- | :--- | :--- |
| **Visual Grid Overlay** | Draw the detected grid lines (for a given `cell_size` and `offset`) directly on top of the original input image. | This is the most important and immediate sanity check. It allows you to instantly see if the grid aligns with the visual blocks, revealing errors in either the cell size or the offset. The conversation shows this was the primary way you and the AI judged success or failure at each step. |
| **Boundary-Edge Scoring** | For a candidate grid, count how many pixels fall directly on the grid lines that have a different color than their immediate neighbor across the line. The grid with the highest score (most edge transitions) is the best fit. | This is an excellent quantitative method to find the optimal grid offset once you have a good guess for the cell size. It's more robust than simple manual alignment. |
| **Edge-Projection & Peak Spacing** | 1. Create a 1D array (a projection) for the X and Y axes by summing the edge-detected pixels along each row and column. <br>2. Find the positions of the peaks in these 1D arrays. <br>3. Calculate the distances between consecutive peaks. The most common distance is the true cell size. | This proved to be the most robust test/method for finding the **true `cell_size`**, even with noise, anti-aliasing, and non-integer scaling. It directly measures the repeating pattern of the artwork's structure, making it superior to simple run-length counting. |
| **Intra-Block Variance** | As a tie-breaker, calculate the color variance *inside* the blocks for several candidate grid sizes. The correct grid size should have the lowest internal variance because it aligns best with the flat-color areas. | This test is perfect for distinguishing between a correct grid size and its sub-multiples (e.g., telling a 32px grid from an 8px grid), as the smaller, incorrect grid would have higher variance by cutting through intended color blocks. |

## 2. Tests for Artifacts and Stray Pixels

After snapping the colors to the grid, the next challenge was dealing with small errors, or "jaggies."

| Test Name | How it Works (The "Test") | What it's Good For |
| :--- | :--- | :--- |
| **Logical Neighbor Majority Vote** | After snapping, iterate over the output image on a *logical* (block-by-block) level. For each block, check the color of its 4 or 8 neighbors. If the block's color is a significant outlier compared to its neighbors, replace it with the majority color. | This is a targeted test and fix for isolated, single-block errors ("stray pixels" or "holes") inside larger uniform areas like the smiley's eyes or outline. It smooths out errors without affecting the overall structure. |
| **High-Resolution Difference Overlay** | 1. Upscale the final pixel-art output back to the original's resolution using nearest-neighbor scaling. <br>2. Compute a per-pixel difference between this upscaled image and the original source image. <br>3. Render this difference as a high-contrast image (e.g., magenta on black). | This became the ultimate **objective test for accuracy**. It visually flags *any* deviation, no matter how small. It was key to proving that a fix didn't work and helped identify exactly where the algorithm was failing (e.g., flattening details vs. removing actual strays). |
| **Block-Difference Stray Filter** | This is a more advanced version of the difference overlay. It flags a block as a "stray" only if the *average difference* between the original block and the snapped block is above a certain threshold. | This test is a refinement that cleverly distinguishes between **expected differences** (like removed anti-aliasing at an edge) and **unexpected differences** (a truly out-of-place block). It stops the algorithm from "fixing" parts of the image that should be preserved. |

## Summary: The Most Valuable Tests

From your development process, the two most powerful and conclusive tests that evolved were:

1. **Edge-Projection Peak Analysis:** This was the breakthrough for accurately *detecting* the non-uniform grid dimensions without making false assumptions about a fixed `cell_size`.
2. **High-Resolution Difference Overlay:** This was the breakthrough for objectively *verifying* the final output, moving beyond subjective "it looks wrong" to a precise, data-driven check that could pinpoint the location and nature of any remaining artifacts.

## Implementation Notes

- Visual Grid Overlay should be the first test implemented in any new grid detection tool
- Edge-Projection Peak Analysis is the most reliable automated method for cell size detection
- High-Resolution Difference Overlay is essential for objective validation of results
- All tests should be modular and reusable across different input images and grid configurations
