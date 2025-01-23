# CUDA Kernels for ML
Currently a work-in-progress repo containing CUDA kernels optimized for various common machine learning algorithms.

## SGEMM (single-precision general matrix-multiplication)
Kernel source code lives under `src/gemm/kernels`

Kernels 0 through 4 are written in pure CUDA C++, but following kernels use either NVIDIA's CUTLASS/CuTe or cuBLAS libraries, as copying/tiling/MMA indices quickly become too complex to manually track.

*Note: Kernels 0 through 4 have tons of hard-coded values, especially for kernel launch parameters. For the sake of simplicity, there is also no bounds checking for kernels 2 through 4, so they are no guarantees for correctness when using non-square `M`, `N`, `K` dimensions that aren't multiples of 64 (i.e. 64, 128, 192, etc.).*

### Kernel 0: [Naive](src/gemm/kernel/0_naive.cuh)

### Kernel 1: [SMEM Blocktiling](src/gemm/kernel/1_shared_mem.cuh)

### Kernel 2: [SMEM Blocktiling + 1D Threadtiling](src/gemm/kernel/2_onedim_blocktile.cuh)

### Kernel 3: [SMEM Blocktiling + 2D Threadtiling](src/gemm/kernel/3_twodim_blocktile.cuh)

### Kernel 4: [SMEM Blocktiling + 2D Threadtiling + Vectorized Transactions](src/gemm/kernel/4_twodim_blocktile_vectorized.cuh)