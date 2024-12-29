#include <iostream>
#include <assert.h>
#include "../../../utils.cuh"

#define SIZE 1024

template <int block_M, int block_N, int block_K, int thread_M>
__global__ void oneDimBlocktile(int M, int N, int K, float* A, float* B, float* C) {
    // TODO: add assertions for kernel launch parameters
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float A_tile[block_M * block_K];
    __shared__ float B_tile[block_K * block_N];
    int tiles = CEIL_DIV(K, block_K);
    float C_res[thread_M] = {0.0f};  // store intermed. C column & cached B value in TMEM
    for (int tile = 0; tile < tiles; ++tile) {
        // TODO: Simplify this complicated GMEM A indexing logic
        A_tile[threadIdx.x * block_K + threadIdx.y] = A[(block_M * blockIdx.y + threadIdx.x) * K + (block_K * tile + threadIdx.y)];
        B_tile[threadIdx.y * block_N + threadIdx.x] = B[(block_K * tile + threadIdx.y) * N + global_idx];
        __syncthreads();
        for (int k = 0; k < block_K; ++k) {
            float B_tmp = B_tile[k * block_N + threadIdx.x];
            for (int thread_row = 0; thread_row < thread_M; ++thread_row) {
                C_res[thread_row] += A_tile[(thread_M * threadIdx.y + thread_row) * block_K + k] * B_tmp;
            }
        }
        __syncthreads();
    }      for (int thread_row = 0; thread_row < thread_M; ++thread_row) {
        C[(thread_M * global_idy + thread_row) * N + global_idx] = C_res[thread_row];
    }
}

int main(int argc, char* argv[]) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float *d_A, *d_B, *d_C;
    auto A = new float[SIZE * SIZE];
    auto B = new float[SIZE * SIZE];
    fill_random(A, SIZE * SIZE, 0.1, 0.2);
    fill_random(B, SIZE * SIZE, 0.1, 0.2);
    
    auto C = new float[SIZE * SIZE];
    auto A_size = sizeof(float) * SIZE * SIZE;
    auto B_size = sizeof(float) * SIZE * SIZE;
    auto C_size = sizeof(float) * SIZE * SIZE;
    cudaMallocAsync((void**)&d_A, A_size, stream);
    cudaMallocAsync((void**)&d_B, B_size, stream);
    cudaMallocAsync((void**)&d_C, C_size, stream);
    cudaMemcpyAsync(d_A, A, A_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, B_size, cudaMemcpyHostToDevice, stream);
    
    constexpr int block_M = 64;
    constexpr int block_N = 64;
    constexpr int block_K = 8;
    constexpr int thread_M = 8;
    dim3 gridDim(CEIL_DIV(SIZE, block_M), CEIL_DIV(SIZE, block_N));
    dim3 blockDim(block_N, block_M / thread_M);
    oneDimBlocktile<block_M, block_N, block_K, thread_M><<<gridDim, blockDim, 0, stream>>>(SIZE, SIZE, SIZE, d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    
    cudaMemcpyAsync(C, d_C, C_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    auto C_ref = new float[SIZE * SIZE];
    reference_gemm(SIZE, SIZE, SIZE, A, B, C_ref);
    compare_matrices(C, C_ref, SIZE, SIZE, 0.00001);
    
    // print(C, SIZE, SIZE);
    // print(C_ref, SIZE, SIZE);
    
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    cudaStreamDestroy(stream);
    return 0;
}
