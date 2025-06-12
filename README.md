# Pixel Perfecter

This repository is an active research and development log focused on converting AI-generated pseudo–pixel art into true, grid-aligned pixel art. It documents a series of algorithmic and manual experiments to reverse-engineer logical sprite grids from noisy, diffusion-based image outputs.

This is not a polished tool. It is a structured workspace for prototyping solutions, gathering insights, and producing material for blog posts, interviews, and portfolio documentation.

---

## Objective

Many AI-generated "pixel art" images suggest grid-like structure but lack true alignment or discrete pixel regions. This project aims to recover or reconstruct that structure through:

- Visual block size detection (e.g. modal run-length histograms)
- Grid offset optimization (to find true origin of implied tiles)
- Block snapping using mode color per region
- Diagnostic image diffs and structure validation tools
- Dataset generation and CNN model training (planned)

---

## Repository Structure

```plaintext
pixel-perfecter/
├── src/                  # All current scripts, experiments, and tooling
├── notes/                # Development journal, findings, failures, ideas
├── tests/                # Planned input/output test assets, CNN evaluations
├── legacy_attempt/       # Archived older tools and failed pipelines
├── README.md             # This file
└── requirements.txt      # Dependencies for all tools
```

---

## Project Philosophy

* Manual verification is the ground truth. Visual alignment beats numeric guesswork.
* Diffusion artifacts must be interpreted structurally, not just semantically.
* Broken experiments are just as valuable as successful ones—everything gets logged.
* This repo doubles as a workspace and documentation hub for future use and sharing.

---

## Challenges

* AI-generated pixel art often looks like good pixel art zoomed out but actually has non-uniform grids, misaligned stray pixels, irregular relationship between wigth and height of grid or individual blocks etc. This becomes problematic when trying to brute force a solution with AI-assisted coding as the AI tends to assume regular grid, perfect alignments and so one. 
* Another challenge is that the AI tends to assume success when the scripts run without console errors, but is unable to check the output visually for adherence to the desired goal. This can become pretty hilarious with the agentic AI creating self-congratulatory 'Mission Accomplished!' md files and scripts while the output is total garbage. 

## Notes System

All key work is tracked in `/notes/`. Each file serves a focused role:

* `logbook.md` – chronological dev notes and daily/weekly check-ins
* `findings.md` – working truths: things that are known to work or be reliable
* `failures.md` – approaches that didn't work and why
* `insights.md` – broader observations or design-level conclusions
* `wishlist.md` – tooling, experiments, or features to build later

This system prioritizes clarity, traceability, and reuse.

---

## Scripts and Tooling

As the project evolves, the following will be developed in `src/`:

* Grid block detectors using histograms and offset scorers
* Color snapping and sprite extraction tools
* Sprite visualizers and diff tools
* Dataset generators for CNN training
* Lightweight training scaffolds for grid inference models

No single monolithic solution is assumed—tools will be modular and task-specific.

---

## Current Status

This repository was repurposed in June 2025. Earlier tools based on FFT and heuristic segmentation have been moved to `/legacy_attempt/`.

The current version is focused on visual grid alignment, mode-color block snapping, and prepping for CNN-based experiments. Scripts are experimental, not yet production-grade.

---

## License

MIT License — see `LICENSE` for details.