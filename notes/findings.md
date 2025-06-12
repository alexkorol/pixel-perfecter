# Key Findings

### Note on Grid Detection (Updated 2025-06-11)
Early iterations assumed standard logical resolutions (e.g., 32×32, 16×16), but testing showed that diffusion-generated pseudo-pixel art varies significantly in implied grid structure.  
All future reconstructions should begin with dynamic grid detection. Assumptions about grid size must be verified per image.