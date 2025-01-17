#include <assert.h>
#include "../../utils.cuh"

template <int block_M, int block_N, int block_K, int thread_M, int thread_N>
__global__ void twodim_blocktile_vectorized(int M, int N, int K, float* A, float* B, float* C) {
    assert(blockDim.x * blockDim.y == block_M * block_N / (thread_M * thread_N));
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_idy = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float A_tile[block_K * block_M];
    __shared__ float B_tile[block_K * block_N];
    const int tiles = CEIL_DIV(K, block_K);
    float res[thread_M * thread_N] = {0.0};
    float A_register[thread_M] = {0.0};
    float B_register[thread_N] = {0.0};
    
    constexpr int A_load_tiles = (block_M * block_K / 4) / (block_M * block_N / (thread_M * thread_N));
    constexpr int B_load_tiles = (block_K * block_N / 4) / (block_M * block_N / (thread_M * thread_N));
    
    // how many threads needed to cover columns?
    constexpr int A_load_cols = block_K / 4;  // 2 for default params
    // how many rows can entire threadblock cover in one load_tile iteration?
    constexpr int A_load_rows = block_M / (A_load_cols); // 32 for default params
    // mapping coordinate (8, 8) -> (32, 2) for default params
    // this mapping would be significantly easier w/ NVIDIA's CUTLASS/CUTE, but will leave this for later kernels
    const int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int A_load_x = linear_idx % A_load_cols;
    const int A_load_y = linear_idx / A_load_cols;
    
    for (int tile = 0; tile < tiles; ++tile) {
        for (int load_tile = 0; load_tile < A_load_tiles; ++load_tile) {
            float4 vector_load = reinterpret_cast<float4*>(
                &A[(blockIdx.y * block_M + (load_tile * A_load_rows + A_load_y)) * K + (tile * block_K + (A_load_x * 4))]
            )[0];
            A_tile[load_tile * block_M + linear_idx] = vector_load.x;
            A_tile[(load_tile + 1) * block_M + linear_idx] = vector_load.y;
            A_tile[(load_tile + 2) * block_M + linear_idx] = vector_load.z;
            A_tile[(load_tile + 3) * block_M + linear_idx] = vector_load.w;
        }
        for (int load_tile = 0; load_tile < B_load_tiles; ++load_tile) {
            reinterpret_cast<float4*>(&B_tile[ ])[0] = 
                reinterpret_cast<float4*>(
                    &B[(tile * block_K + threadIdx.y) * N + (blockIdx.x * block_N + (load_tile * load_K + threadIdx.x))]
                )[0];
        }
        __syncthreads();

    }
}

void launch_twodim_blocktile_vectorized(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    constexpr int thread_N = 8;
    assert(block_M / thread_M == block_N / thread_N);
    assert(block_M / thread_M == block_K);

    dim3 blockDim(block_M / thread_M, block_N / thread_N);
    dim3 gridDim(CEIL_DIV(M, block_M), CEIL_DIV(N, block_N));
    twodim_blocktile_vectorized<block_M, block_N, block_K, thread_M, thread_N>
        <<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK(cudaGetLastError());
}