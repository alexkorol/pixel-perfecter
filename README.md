# Pixel Perfecter

Pixel Perfecter is a Python-first toolkit for turning diffusion-flavoured "pixel art" into clean, grid-aligned sprites. The codebase now revolves around a reusable reconstruction module, a Qt GUI for interactive tuning, and a batch CLI that mirrors the GUI settings for large runs. Inspiration comes from unfake.js (rich controls, great UX) and Proper Pixel Art (clear documentation, math-first analysis); the goal is to fuse clarity with interactivity in one repository.

---

## Components at a Glance

- **Core pipeline (`pixel_perfecter/reconstructor.py`)** – Estimates pixel grids, reconstructs colour cells, and emits diagnostics that separate halo noise from core differences.
- **Interactive GUI (`python -m pixel_perfecter.gui`)** – PySide6 application with drag-and-drop import, live previews, optional ML hints, and structured feedback logging.
- **Batch CLI (`python -m pixel_perfecter.cli`)** – Processes files or directories, writes reconstructions plus overlays, and records metrics to CSV.
- **Optional ML helpers (`ml/`)** – Lightweight PyTorch models that suggest grid parameters; both GUI and CLI auto-detect their availability.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# or: source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

The `input/` and `output/` folders are git-ignored so you can freely drop working images and generated results there.

---

## Batch CLI

The batch interface lives under `pixel_perfecter.cli` and wraps the reconstruction pipeline with sensible defaults.

```bash
# Basic run on every PNG / JPEG / BMP / WebP in ./input
python -m pixel_perfecter.cli input --output-dir output

# Apply ML suggestions, walk directories recursively, and split overlays
python -m pixel_perfecter.cli input extra/*.png \
    --mode global \
    --ml \
    --recursive \
    --overlay-dir output/overlays
```

### Key options

| Flag | Description |
| ---- | ----------- |
| `inputs...` | Files or directories to process. Directories are scanned (optionally recursively) for supported image types. |
| `-o, --output-dir PATH` | Destination for reconstructed sprites (default: `./output`). |
| `--overlay-dir PATH` | Optional directory for validation overlays (defaults to the output dir). |
| `--mode {global,adaptive}` | Choose the grid strategy. Adaptive mode fits per-region grids when a single lattice fails. |
| `--overlay-mode {combined,raw,halo_suppressed,core_only}` | Select which diagnostic composite to export. |
| `--skip-overlays` | Do not write overlay images. |
| `--ml` | Enable ML suggestions if `ml/` checkpoints are present (ignored in adaptive mode). |
| `--debug` | Emit verbose logging from the reconstruction pipeline. |
| `--grid-debug` / `--grid-debug-dir PATH` | Export grid detection plots (defaults to `output/grid_debug/`). |
| `--recursive` | Walk directory inputs recursively. |
| `--metrics-path PATH` / `--no-metrics` | Control where the metrics CSV lands (defaults to `output/metrics.csv`). |

Each run produces:

- `<output>/<name>_reconstructed.png` – the snapped sprite.
- `<output>/<name>_validation.png` (unless skipped) – the chosen overlay variant highlighting halo versus core differences.
- `metrics.csv` – per-image statistics (`percent_diff_core`, `percent_diff_halo`, `grid_ratio`, warnings, ML usage, output paths).
- `grid_debug/` (when enabled) – edge projections and autocorrelation plots captured during grid inference.

---

## GUI

Launch the PySide6 interface with:

```bash
python -m pixel_perfecter.gui
```

Highlights:

- Drag-and-drop loading, live recomputation when cell size or offsets change, and instant previews of multiple overlay styles.
- Optional "Adaptive per-region grid" mode for locally warped assets.
- Halo-aware metrics (core / halo / outside) shown alongside the preview, mirroring the CLI CSV fields.
- ML suggestions (when `ml/` checkpoints exist) displayed in a sidebar; the GUI gracefully degrades if the models are absent.
- Feedback logging to `notes/feedback_log.csv`, making it easy to curate success and failure cases while testing.

---

## Pipeline Overview

1. **Pre-processing** – optional alpha binarisation and halo-aware edge detection.
2. **Grid detection** – autocorrelation and Sobel projections estimate dominant cell sizes and offsets; adaptive mode splits the image into regions when confidence is low.
3. **Cell reconstruction** – each grid cell collapses to a modal colour taken from its interior, limiting halo contamination.
4. **Diagnostics** – reconstructions are upscaled, overlays are rendered in several variants, and difference statistics (core / halo / outside) are recorded for manual review.
5. **Optional refinement** – ML suggestions (if enabled) evaluate alternate grid parameters and keep the best-scoring candidate.

The CLI and GUI both use this pipeline, so switching between interactive and automated workflows is lossless.

---

## Repository Layout

```
pixel-perfecter/
├─ pixel_perfecter/        # Python package: reconstructor, GUI launcher, CLI
├─ src/                    # Legacy shims kept for backward compatibility
├─ ml/                     # Optional ML helpers (dataset prep, inference, checkpoints)
├─ input/                  # Drop source images here (not tracked)
├─ output/                 # Default destination for reconstructions (not tracked)
├─ notes/                  # Design docs, logbooks, curated examples
├─ archive/                # Earlier experiments and reference scripts
├─ requirements.txt        # Runtime dependencies
└─ README.md               # Project overview
```

---

## Roadmap

- Harden CLI smoke tests with additional fixtures and edge-case assertions.
- Migrate remaining development scripts from `src/` into the `pixel_perfecter` namespace or archive them.
- Document algorithmic trade-offs (halo handling, adaptive grids) in `notes/` with side-by-side visuals.
- Package optional ML checkpoints for easier distribution and surface confidence metrics more prominently in the GUI and CLI.

---

## License

MIT License – see `LICENSE` for full text.
