#include "../../utils.h"

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int TM, int TN, int TK,
          int thrs, int blks>
__global__ void __launch_bounds__(thrs, blks) warptile_no_cg(
    int M, int N, int K, float* A, float* B, float* C
) {
    extern __shared__ float shared_A[];
    extern __shared__ float shared_B[];
    float reg_A[TM * TK] = {0.0};
    float reg_B[TK * TN] = {0.0};
    float reg_C[TM * TN] = {0.0};
    
}

__host__ inline void launch_warptile_no_cg(
    int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream
) {
    // TODO: conditional compile for each arch
    // these are for sm_86
    constexpr int regfile_bytes_sm = 65536;
    constexpr int smem_bytes_sm = 102400;

    // assuming a NN row-major GEMM problem
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 4;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int TK = 4;

    // block-level
    constexpr int warps = (BM / WM) * (BN / WN);
    constexpr int threads = warps * 32;
    constexpr int registers = threads * ((TM * TK) + (TK * TN) + (TM * TN));  // rough estimate
    constexpr int smem_bytes = sizeof(float) * ((BM * BK) + (BK * BN));
    constexpr int min_blocks = std::min(smem_bytes_sm / smem_bytes, regfile_bytes_sm / registers);

    // warp-level
    constexpr int warptile_size = WM * WN;
    constexpr int threadtile_size = warptile_size / 32;
    constexpr int threadtiles = threadtile_size / (TM * TN);

    static_assert(threads <= 1024);
    static_assert(!(BM % WM));
    static_assert(!(BN % WN));
    static_assert(!(BK % WK));
    static_assert(!(WM % TM));
    static_assert(!(WN % TN));
    static_assert(!(WK % TK));

    dim3 grid_dim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    constexpr dim3 block_dim(threads);
    warptile_no_cg<BM, BN, BK,
                   WM, WN, WK,
                   TM, TN, TK,
                   threads, min_blocks>
                   <<<grid_dim, block_dim, (BM * BK) + (BK * BN), stream>>>
                   (M, N, K, A, B, C);
    
    /*
        (BM, BN, BK) = (128, 128, 8)
        (WM, WN, WK) = (64, 64, 4)
        (TM, TN, TK) = (4, 4, 2)

        sA = 128 * 8 = 1024
        sB = 8 * 128 = 1024
        ~ 2048 fp32 / CTA = 8.192 kB / CTA
        100 kB / SM --> 100 / 8.192 = 12 CTA / SM

        rA = 4 * 2 = 8
        rB = 2 * 4 = 8
        rC = 4 * 4 = 16
        ~ 32 register / thread
        
        (128 / 64) * (128 / 64) = 4 warp / CTA
        4 * 32 = 128 thread / CTA
        128 * 32 = 4096 register / CTA
        65536 / 4096 = 16 CTA / SM

        shared memory is the bottleneck
    */

    /*
        how does WK / TK affect occupancy?
        suppose WK / TK = 4 --> threads perform 4 (unrolled) loop iters. over WK for MMA --> lower register pressure
        suppose WK / TK = 1 --> higher register pressure to store (TM * TK) + (TK * TN) fp32
        we can afford higher register pressure, since SMEM is currently bottleneck

        what about BK / WK?
        suppose BK / WK = 2 --> warps perform 2 iters. over BK
        a higher WK could mean lower register pressure if TK is held constant

        suppose all other params. held constant, but TK = 4
        (4 * 4) + (4 * 4) + (4 * 4) = 48 register / thread

        128 thread / CTA * 48 register / thread = 6144 register / CTA
        65536 / 6144 = 10 CTA / SM

        registerfile is now the bottleneck --> if registerfile bottleneck results in less CTAs per SM, what's the benefit?
        each thread covers WK in one iteration, higher compute intensity / thread + higher ILP on CUDA cores
    */
}