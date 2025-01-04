#include "../../utils.cuh"

template <int block_M, int block_N, int block_K, int thread_M, int thread_N>
__global__ void twodim_blocktile(int M, int N, int K, float* A, float* B, float* C) {
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float A_tile[block_M * block_K];
    __shared__ float B_tile[block_K * block_N];
    const int tiles = CEIL_DIV(K, block_K);
    float res[thread_M * thread_N] = {0.0};
    float A_register[thread_N] = {0.0};
    float B_register[thread_M] = {0.0};
    for (int tile = 0; tile < tiles; ++tile) {
        const int load_K = (blockDim.x * blockDim.y) / block_K;
        // TODO: Fill in tile load indexing (GMEM -> SMEM)
        for (int load_tile = 0; load_tile < CEIL_DIV(block_M, load_K); ++load_tile) {
            A_tile[ ] = A[ ];
        }
        for (int load_tile = 0; load_tile < CEIL_DIV(block_N, load_K); ++load_tile) {
            B_tile[ ] = A[ ];
        }
    }
}

void launch_twodim_blocktile(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    constexpr int thread_N = 8;
    twodim_blocktile<block_M, block_N, block_K, thread_M, thread_N>;
}