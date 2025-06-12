# Key Findings

### Note on Grid Detection (Updated 2025-06-11)
Early iterations assumed standard logical resolutions (e.g., 32×32, 16×16), but testing showed that diffusion-generated pseudo-pixel art varies significantly in implied grid structure.  
All future reconstructions should begin with dynamic grid detection. Assumptions about grid size must be verified per image.

### Non-uniform grid and non-square pixels. 
The key finding is that we must start from the assumption that the grid in the ai generated pixel art is going to be non-uniform and the pixels are going to be non-square and with individual offset from the overall grid. 

