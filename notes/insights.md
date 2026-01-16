# Development Insights

- Most of the work is verifying grid assumptions visually
- AI tools don't reason in terms of implied pixels unless forced
- A debug grid overlay tool would help rapidly iterate on alignment

### A Unified Pipeline Beats a Medley of Tools

The most effective and maintainable approach is not to have numerous small, specialized scripts, but to build a **single, robust, and well-defined pipeline** (e.g., a `reconstructor.run()` method).

This central pipeline should then be driven by a single, focused validation script that is responsible for generating all necessary debug artifacts (grid overlays, difference maps, etc.). This architecture:

1.  Enforces a consistent, repeatable process.
2.  Makes it trivial to evaluate the impact of any single change.
3.  Keeps the core logic (`src`) clean from testing and validation code.

### AI tools don't reason in terms of implied pixels unless forced. 

AI tends to be very lazy in its assumptions and will assume "best case scenarios" when tackling complex problems, resulting in incorrect output. 

### Agentic AI assistants will be "confidently mistaken" and will manipulate test results to avoid confronting difficult problems.

It will assume success simply because the scripts it created ran, but will be oblivious to obviously bad output of these scripts. Then it will prematurely celebrate success by making outputs in the console or md files. Sometimes it will knowingly hardcode the test instead of using test as a tool to debug.

---

## External References & Related Projects

### 1. proper-pixel-art by Kenneth J. Allen (Python)
**GitHub:** https://github.com/KennethJAllen/proper-pixel-art

A mathematically rigorous approach to recovering true pixel art from AI-generated images. Well-documented and worth studying both for implementation and the thinking process behind it.

**Core Algorithm:**
1. Preprocessing - trim edges, threshold alpha channel
2. 2x nearest-neighbor upscaling to clarify pixel boundaries
3. Canny edge detection to identify pixel boundaries
4. Morphological closing to fill small gaps in edges
5. Probabilistic Hough transform to extract grid lines
6. Median spacing of clustered lines determines cell dimensions
7. K-means clustering for color quantization
8. Most-common color within each mesh cell → output pixel

**Key Mathematical Concepts:**
- Hough Transform - detects linear features via voting in parameter space
- Morphological closing - dilation then erosion to bridge gaps
- K-means clustering - palette reduction
- Median filtering - robust outlier rejection for grid spacing

**Strengths:** Deterministic, no ML needed, works well on axis-aligned grids
**Limitations:** Struggles with rounded objects and non-uniform grids

**Why study this:** Clean mathematical thinking, well-structured code, good reference for understanding the problem from first principles.

---

### 2. unfake.js by jenissimo (JavaScript/Browser)
**GitHub:** https://github.com/jenissimo/unfake.js
**Demo:** Available on itch.io

A browser-based tool with dual-mode support (pixel art cleanup + vectorization). Good for quick tests and understanding user expectations.

**Pixel Art Pipeline:**
1. Scale detection (runs-based or edge-aware algorithms)
2. Content-aware downscaling (dominant, median, or adaptive)
3. Grid alignment / auto-crop
4. Color quantization via image-q library
5. Optional morphological and jaggy cleanup

**Vectorization Pipeline:**
- Wraps imagetracer.js with bilateral/median blur preprocessing
- Pre-trace quantization for cleaner vector shapes

**Key Features:**
- Real-time parameter adjustment via Tweakpane UI
- Built-in magnification and palette editing
- Drag-and-drop, paste from clipboard
- Outputs PNG or SVG

**Also Available:**
- ComfyUI workflow (ComfyUI-Unfake-Pixels)
- Python port (unfake.py) with Rust acceleration

**Why study this:** Good UX patterns, browser-based approach, handles both pixel and vector output.

---

### Comparison Notes

| Aspect | proper-pixel-art | unfake.js |
|--------|------------------|-----------|
| Language | Python (PIL, OpenCV, NumPy) | JavaScript (OpenCV.js, image-q) |
| Approach | Hough transform + median spacing | Runs/edge-based scale detection |
| Grid detection | Line detection → median clustering | Content-aware algorithms |
| Color reduction | K-means | image-q library |
| Strengths | Mathematical rigor, deterministic | UX, real-time preview, dual output |

**Note:** Our project (pixel-perfecter) previously referenced "Astropulse/pixeldetector" in ml/heuristics.py - the proper-pixel-art project is the actual reference we should be learning from. 

