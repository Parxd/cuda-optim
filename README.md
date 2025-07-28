# Optimzing CUDA Kernels
Currently a work-in-progress repo containing CUDA kernels optimized for various common machine learning algorithms.

## SGEMM (single-precision general matrix-multiplication)
Kernel source code lives under `src/gemm/kernels`

Kernels 0 through 5 are written in pure CUDA C++, but following kernels use either NVIDIA's CUTLASS/CuTe or cuBLAS libraries, as copying/tiling/MMA indices quickly become too complex to manually track.

*Note: Kernels 0 through 4 have lots of hard-coded kernel launch parameters and messy code, as they were mostly for demonstration and learning general good GEMM concepts (block/thread-tiling). As a result, there is also no bounds checking for kernels 2 through 4, so they are no guarantees for correctness when using non-square `M`, `N`, `K` dimensions that aren't multiples of `64` (i.e. =/= 64, 128, 192, etc.).*

Kernels 5 and above have far cleaner (and performant) code.
The general approach used for these kernels (CTA-tiling, warp-tiling, thread-tiling) can be visualized with this diagram from NVIDIA's CUTLASS...
![img](res/3.png)

The following performance tests were run on my RTX 3070 Mobile for `M = N = K = 512`.

- Kernel 0: [Naive](src/gemm/kernel/0_naive.cuh)
    - Time: 
    - G                           s: 

- Kernel 1: [SMEM Blocktiling](src/gemm/kernel/1_shared_mem.cuh)
    - Time: 
    - GFLOPs: 

- Kernel 2: [SMEM Blocktiling + 1D Threadtiling](src/gemm/kernel/2_onedim_blocktile.cuh)
    - Time: 
    - GFLOPs: 

- Kernel 3: [SMEM Blocktiling + 2D Threadtiling](src/gemm/kernel/3_twodim_blocktile.cuh)
    - Time: 
    - GFLOPs: 

- Kernel 4: [SMEM Blocktiling + Threadtiling + Vectorized Transactions](src/gemm/kernel/4_twodim_blocktile_vectorized.cuh)
    - Time: 
    - GFLOPs: 
    
- Kernel 5: [SMEM Blocktiling + Warptiling + Threadtiling + Vectorized Transactions](src/gemm/kernel/5_warptile.cuh)
    - Time:
    - GFLOPs: 