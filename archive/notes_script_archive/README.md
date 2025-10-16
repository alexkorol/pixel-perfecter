# Pixel Perfecter Script Archive

This directory captures every major reconstruction approach checked into the repository so far. Each subfolder contains the exact scripts as they existed in the referenced commit so you can run or diff them side-by-side without juggling `git checkout`.

- `4332ac1_initial_fft/` — Commit `4332ac1` (“Initial commit: Working FFT-based scale detection with harmonic analysis”).  
  Early pipeline built from individual modules (`analysis.py`, `reconstruction.py`, etc.). Uses FFT-based scale estimation plus heuristic segmentation. Heavy reliance on SciPy and manual tuning; no automatic reconstruction runner.

- `cf36e7d_transition/` — Commit `cf36e7d` (“Archived first implementation. Created clean structure…”).  
  Introduced `pixel_perfecter.py` + `scale_detection.py` and a small Flask front-end (`web_main.py`). Still FFT oriented but wrapped as a scriptable module instead of ad-hoc notebooks.

- `05e412b_first_pipeline/` — Commit `05e412b` (“Implemented complete pixel art reconstruction pipeline”).  
  First end-to-end CLI with `pixel_reconstructor.py`, analysis utilities (`analyze_images.py`, `visual_analysis.py`), and regression harness (`test_all_images.py`). Grid detection delegated to `GridDetectionTests`; modal color snap without OpenCV edge metrics.

- `a7f30e0_refactor/` — Commit `a7f30e0` (“Refactor pixel art reconstruction pipeline and remove legacy testing scripts”).  
  Rewrites the pipeline to use OpenCV edges + autocorrelation for cell size, plus neighbor majority refinement. Adds `find_peaks`-based detection and consolidates state management.

- `a0c74e1_harmonics/` — Commit `a0c74e1` (“Enhance pixel art reconstruction pipeline…”).  
  Expands the refactor with harmonic grouping, validation overlays, and Matplotlib grid debug dumps. Switches entry point to path-based constructor (`PixelArtReconstructor(image_path)`).

- `544777d_roocode_success/` — Commit `544777d` (“Using RooCode got to where all four test images pass…”).  
  Adds the multi-metric cell-size refinement (edge alignment, block variance, reconstruction diff) that was stable for the four benchmark inputs.

Current `HEAD` (`2254722`) is the version you just tested locally: `src/pixel_reconstructor.py` with further metric tuning and automated warnings. Refer to it directly in `src/` for comparison.

The historical `legacy_attempt/` directory in the repo already contains the pre-archive web experiments; keep using those if you want to revisit the very first wave of ideas.
