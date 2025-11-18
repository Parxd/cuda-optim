#include <stdio.h>
#include "../../utils.h"

#define WIDTH 4  // pack 4 fp32

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int TM, int TN, int TK,
          int WIM, int WIN,
          int thrs, int blks>
__global__ void __launch_bounds__(thrs, blks) warptile_no_cg(
    int M, int N, int K, float* A, float* B, float* C
) {
    __shared__ float shared_A[BM * BK];
    __shared__ float shared_B[BK * BN];
    float reg_A[TM * TK * WIM]  = {0.0};
    float reg_B[TK * TN * WIN] = {0.0};
    float reg_C[TM * TN * WIM * WIN] = {0.0};

    constexpr int gA_lds = (BM * BK) / thrs / WIDTH;
    constexpr int gA_ld_rows = (thrs * WIDTH) / BK;
    constexpr int gB_lds = (BK * BN) / thrs / WIDTH;
    constexpr int gB_ld_rows = thrs / (BN / WIDTH);

    int thread_row_A = threadIdx.x / (BK / WIDTH);
    int thread_col_A = threadIdx.x % (BK / WIDTH);
    // for 2nd loading schema
    int thr_row_A = (threadIdx.x / BK) * WIDTH;
    int thr_col_A = threadIdx.x % BK;

    int thread_row_B = threadIdx.x / (BN / WIDTH);
    int thread_col_B = threadIdx.x % (BN / WIDTH);

    float* global_A = &A[blockIdx.y * BM * K];
    float* global_B = &B[blockIdx.x * BN];

    int tiles = K / BK;
    for (int tile = 0; tile < tiles; ++tile) {
        /*
        1st loading schema: ld.global.v4.f32 -> st.shared.f32
        benchmarking shows higher relative bandwidth on lower BK = 16 sizes (highest absolute bandwidth)
        */
#pragma unroll
        for (int ld = 0; ld < gA_lds; ++ld) {
            float* global_thread = &global_A[(ld * gA_ld_rows + thread_row_A) * K + (tile * BK + (thread_col_A * 4))];
            auto load = reinterpret_cast<float4*>(global_thread)[0];
            shared_A[(thread_col_A * WIDTH) * BM + (ld * gA_ld_rows + thread_row_A)] = load.x;
            shared_A[(thread_col_A * WIDTH + 1) * BM + (ld * gA_ld_rows + thread_row_A)] = load.y;
            shared_A[(thread_col_A * WIDTH + 2) * BM + (ld * gA_ld_rows + thread_row_A)] = load.z;
            shared_A[(thread_col_A * WIDTH + 3) * BM + (ld * gA_ld_rows + thread_row_A)] = load.w;
        }

        /*
        2nd loading schema: ld.global.f32 -> st.shared.v4.f32
        benchmarking shows higher relative bandwidth on higher BK = 32 sizes
        */
// #pragma unroll
//         for (int ld = 0; ld < gA_lds; ++ld) {
//             float4 load;
//             load.x = global_A[(ld * gA_ld_rows + thr_row_A) * K + (tile * BK + thr_col_A)];
//             load.y = global_A[(ld * gA_ld_rows + thr_row_A + 1) * K + (tile * BK + thr_col_A)];
//             load.z = global_A[(ld * gA_ld_rows + thr_row_A + 2) * K + (tile * BK + thr_col_A)];
//             load.w = global_A[(ld * gA_ld_rows + thr_row_A + 3) * K + (tile * BK + thr_col_A)];
//             reinterpret_cast<float4*>(&shared_A[thr_col_A * BM + (ld * gA_ld_rows + thr_row_A)])[0] = load;
//         }

#pragma unroll
        for (int ld = 0; ld < gB_lds; ++ld) {
            float* global_thread = &global_B[(tile * BK + (ld * gB_ld_rows + thread_row_B)) * N + (thread_col_B * WIDTH)];
            auto load = reinterpret_cast<float4*>(global_thread)[0];
            reinterpret_cast<float4*>(&shared_B[(ld * gB_ld_rows + thread_row_B) * BN + (thread_col_B * WIDTH)])[0] = load;
        }
        __syncthreads();
    }
}

__host__ inline void launch_warptile_no_cg(
    int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream
) {
    // for CC 8.6 ONLY
    // see https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
    constexpr int regfile_size_sm = 65536;  // unit: 32-bit registers
    constexpr int smem_bytes_sm = 102400;
    constexpr int threads_sm = 1536;

    // assuming NN row-major GEMM problem
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 1;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int TK = 1;

    // warp-level
    constexpr int warptile_size = WM * WN;
    constexpr int threadtile_size = warptile_size / 32;
    constexpr int threadtiles = threadtile_size / (TM * TN);
    constexpr int WIM = 2;  // num. "rows" when tiling over warptile
    constexpr int WIN = threadtiles / WIM;  // num. "cols" when tiling over warptile

    // block-level
    constexpr int warps = (BM / WM) * (BN / WN);
    constexpr int threads = warps * 32;
    constexpr int registers = threads * ((TM * TK * WIM) + (TK * TN * WIN) + (TM * TN * WIM * WIN));
    constexpr int smem_bytes = sizeof(float) * ((BM * BK) + (BK * BN));
    constexpr int min_blocks = std::min({
        smem_bytes_sm / smem_bytes,
        regfile_size_sm / registers,
        threads_sm / threads
    });

    static_assert(threads <= 1024);
    static_assert(registers / threads <= 255);  // for CC x.x: max 255 32-bit reg. per thread allowed
    static_assert(!(BM % WM));
    static_assert(!(BN % WN));
    static_assert(!(BK % WK));
    static_assert(!(WM % TM));
    static_assert(!(WN % TN));
    static_assert(!(WK % TK));

    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    constexpr dim3 block_dim(threads);
    warptile_no_cg<BM, BN, BK,
                   WM, WN, WK,
                   TM, TN, TK,
                   WIM, WIN,
                   threads, min_blocks>
                   <<<grid_dim, block_dim, 0, stream>>>
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