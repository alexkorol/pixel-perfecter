# Dev Log: pixel-perfecter v2

## 2025-06-10

- Brainstormed with chatGPT advanced voice. Decided to reboot the pixel perfecter project from a different angle. I will aim to make it an end-to-end project (start from scratch and make a working deployed app) and document all my development and research steps for career development document, interview prep, blogposts etc. I will start by trying to tackle from different angles the task of converting ai generated simulated pixel art (1024 etc images that depict pixel art made in Sora/ChatGPT Imagegen) into actual true-to size pixel art assets. Another approach I will try or at least consider is training a cnn for this task. In this case the other steps could also help in generating a dataset for training. 

## 2025-06-11

**Rebooted project.** Old code moved to `legacy_attempt/`.

- Refocused effort on clean grid alignment from modal block sizes
- Starting new scripts from scratch in `src/`

## 2025-06-11 - Afternoon

**Implemented comprehensive testing suite:**
- Created `src/grid_tests.py` with different methodology tests. In chatgpt o3 the Edge-Projection Peak Analysis and High-Resolution Difference Overlay were the most effective ways to test the output, although the gpt still struggled to stop the failures on these tests. 
- Tested on 4 simulated pixel art images: smiley, pollen, triangle, star. They are the simpler example images I have with more consisten grids

**Results:**
**Smiley & Pollen**: Successfully converted to true 1:1 pixel art
**Triangle**: 2x too large, still has 2-pixel implied blocks (should be 1-pixel)
**Star**: Weird/incorrect reconstruction

**Issues identified:**
- Need better grid size validation
- Edge projection sometimes gives false positives


**Github Copilot with Claude Sonnet 4 getting carried away** Copilot got carried away on a tangent and ended up making a self-congratulatory PROJECT_SUMMARY.py that would output a whole 'Mission Accomplished' schpiel. It keeps thinking that if it makes a running script that the script's output is satisfactory and keeps forgetting that the outputs require human visual review and validation. 

## 2025-06-11 - Late Afternoon

**Strategic Refactoring.** The AI coder's approach became divergent, creating a chaotic sprawl of redundant scripts (`debug_simple.py`, `first_principles.py`, `smart_reconstruction.py`, etc.) instead of converging on a single, effective solution.

**Action Taken:** Issued a firm directive to halt script proliferation. The new plan is to:
1.  **Delete** all redundant experimental/debug scripts.
2.  **Consolidate** all core logic into the existing `src/pixel_reconstructor.py` and `src/grid_tests.py`.
3.  **Unify** the process into a single, clean `PixelArtReconstructor.run()` method.
4.  **Centralize** all validation in a single new script: `tests/validate_pipeline.py`.

Sonnet 4 still gets carried away an creates new files despite explicit instructions against doing so. 

modifying main script to not require any arguments and to just take in all images in /input for easy manual testing.

set up image viewer tool with resample for zooming (non-blurry zoom) for better manual visual validation. 

Sonnet 4 still goes off the rails. Assumes that because script runs without errors the outputs are good. Makes a premature "final_status.md" file. Switching to GPT-4.1 to see if it's possible to stay more on track. 

GPT 4.1 keeps asking too many unnecessary questions (stalling) instead of working agentically and independently. 