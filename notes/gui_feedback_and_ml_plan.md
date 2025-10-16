# Pixel Perfecter Visual Feedback & ML Augmentation Plan

## 1. Interactive GUI Feedback Console

**Goal:** Shorten the experimentation loop by letting us run the reconstruction pipeline, inspect overlays, and record corrections without leaving a visual workspace.

**Core requirements**
- Drag & drop or file picker for any `input/` image.
- Run the current `PixelArtReconstructor` in a worker thread to avoid UI freezes.
- Display three synchronized panes: original, red-diff overlay, reconstructed grid (nearest-neighbour upsampled).
- Surface numeric metrics from `output/metrics.csv` logic (cell size, offsets, % diff). Highlight when they exceed thresholds.
- Manual controls:
  - Override cell size (explicit value, half/double quick actions).
  - Adjust offsets (nudging buttons + numeric entry).
  - Toggle refinement/smoothing steps (for future experiments).
  - Re-run reconstruction instantly on change.
- Feedback capture:
  - “Looks correct” / “Incorrect” toggle.
  - Optional notes field for qualitative remarks.
  - Persist feedback to `notes/feedback_log.csv` with timestamp, overrides, metrics, and user notes.
- Session management:
  - Cache reconstructed frames (to flip between attempts quickly).
  - Allow saving/exporting the current best reconstruction directly to `output/`.

**Stretch items**
- Heatmap overlay showing per-cell disagreement.
- Keyboard shortcuts for frequent overrides.
- Batch mode to iterate through folders with “next/previous” navigation.

**Tech stack**
- PySide6 (Qt for Python) for cross-platform UI and high-DPI friendly rendering.
- QThreads or `QtConcurrent` for pipeline execution.
- Reuse existing numpy→QImage utilities (to be written) for fast display.


## 2. ML-Assisted Grid Estimation Track

**Goal:** Use supervised signals from manual corrections + synthetic data to predict grid parameters or directly output clean sprites, reducing dependence on heuristics.

**Phased approach**
1. **Data collection**
   - Gather corrections from the GUI feedback log.
   - Generate synthetic training images by corrupting ground-truth pixel art (random offsets, scaling, blur/noise).
   - Store ground-truth cell size/offset plus reconstructed diff metrics.
2. **Model experiments**
   - Baseline classifier/regressor predicting `(cell_size, offset_x, offset_y)` from downsampled RGB inputs or edge maps.
   - Optionally frame as segmentation: output gridline probability map or low-res sprite that we compare to reconstruction.
3. **Hybrid inference**
   - Use model to rank candidate grid parameters and feed top-k into the deterministic pipeline.
   - Compare performance against vanilla heuristics using the metrics CSV.
4. **Evaluation**
   - Track accuracy: % of images within ±1 pixel on offsets and exact cell size match.
   - Validate reconstruction diff improvements (target <2%).
5. **Integration**
   - Expose “ML suggestions” in the GUI (list of parameter sets with scores).
   - Allow user to accept/reject suggestions; feed outcomes back as training labels.

**Tech stack considerations**
- PyTorch Lightning (or plain PyTorch) with simple CNN/ResNet blocks to start.
- Augmentation pipeline using Albumentations / torchvision transforms to mimic diffusion artifacts.
- Training scripts under `ml/` with reproducible configs (YAML driven).
- Evaluate deployment footprint (ONNX or TorchScript) for optional runtime integration into GUI.


### Detailed pipeline design (current iteration)

**Datasets**
- `feedback_log.csv` → definitive human labels (`cell_size`, `offset_x`, `offset_y`, optional overrides). Use only rows marked “Looks correct” as positive ground truth.
- `synthetic/` cache → generated sprites with known grids. Persist to disk for reproducibility.
- `input/` recon outputs → on-the-fly pseudo labels via heuristics (treated as weak labels for semi-supervised experiments).

**Pre-processing**
- Convert each image to both RGB and edge maps (Canny). Model input will be 4-channel tensor `[R,G,B,edge]` downscaled to `160×160`.
- Normalise grid parameters by dividing by 256 (clamp to [0,1]).
- Store dataset index as `.jsonl` manifest for reproducibility.

**Model**
- Baseline: lightweight CNN with residual blocks; output regression head producing `(cell_size, offset_x, offset_y)` plus confidence score `c` (0–1). Confidence supervised via label type (1.0 for human, 0.3 for synthetic, 0.1 for weak).
- Loss: weighted MSE on parameters + BCE on confidence.
- Optimiser: AdamW with cosine LR schedule, mixed precision when CUDA available.

**Training loop**
- Support stratified batching by label source.
- Metrics: MAE (in pixels) and accuracy thresholds (|Δcell|≤1 & |Δoffset|≤1). Log with tqdm + TensorBoard.
- Checkpoint best model (lowest validation MAE) + last epoch.

**Inference**
- Expose helper `ml.inference.suggest_parameters(image_path, top_k=3)` returning sorted candidates `(cell_size, offset, confidence)`.
- Hybrid evaluation: pass each candidate into existing pipeline, compute diff% quickly, pick best.
- GUI integration: show suggestions in a dropdown; applying one reruns reconstruction with overrides and records acceptance or rejection.

**Data refresh flow**
1. User reviews images in GUI, marking correct results or overrides.
2. Nightly job (or manual command) runs `python -m ml.prepare_dataset` to regenerate manifest + train/val splits.
3. `python -m ml.grid_estimator.train --config configs/grid_estimator.yaml` fine-tunes model on GPU.
4. Export inference weights (`.pt` + optional TorchScript) consumed by GUI and CLI.

## 3. Open Tasks / TODOs

- [x] Scaffold PySide6 application structure and rendering utilities.
- [x] Build async wrapper around `PixelArtReconstructor` for GUI use.
- [x] Design feedback logging schema (`notes/feedback_log.csv`).
- [ ] Materialise dataset preparation script (manifests, synthetic cache, channel stacking).
- [ ] Implement training config loader + TensorBoard logging.
- [ ] Add inference helper returning ranked parameter suggestions.
- [ ] Surface ML suggestions and accept/reject logging inside the GUI.
- [ ] Define evaluation protocol targeting 98% accurate restorations.
