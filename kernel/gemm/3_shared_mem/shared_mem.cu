#include <iostream>
#include <assert.h>
#include "../../../utils.cuh"

#define SIZE 8192
#define BLOCK 32

__global__ void sharedMem(float* A, float* B, float* C, int size) {
    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_idy = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float A_shared[BLOCK * BLOCK];
    __shared__ float B_shared[BLOCK * BLOCK];
    if (global_idx < size && global_idy < size) {
        float tmp = 0.0;
        const int tiles = CEIL_DIV(size, BLOCK);
        for (int tile = 0; tile < tiles; ++tile) {
            A_shared[threadIdx.y * BLOCK + threadIdx.x] = A[global_idy * size + (tile * BLOCK + threadIdx.x)];
            B_shared[threadIdx.y * BLOCK + threadIdx.x] = B[(tile * BLOCK + threadIdx.y) * size + global_idx];
            __syncthreads();
            for (int k = 0; k < BLOCK; ++k) {
                tmp += A_shared[threadIdx.y * BLOCK + k] * B_shared[k * BLOCK + threadIdx.x];
            }
            __syncthreads();
        }
        C[global_idy * size + global_idx] = tmp;
    }
}

int main(int argc, char* argv[]) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float *d_A, *d_B, *d_C;
    auto A = new float[SIZE * SIZE];
    auto B = new float[SIZE * SIZE];
    fill_random(A, SIZE * SIZE);
    fill_random(B, SIZE *SIZE);

    auto C = new float[SIZE * SIZE];
    auto A_size = sizeof(float) * SIZE * SIZE;
    auto B_size = sizeof(float) * SIZE * SIZE;
    auto C_size = sizeof(float) * SIZE * SIZE;
    cudaMallocAsync((void**)&d_A, A_size, stream);
    cudaMallocAsync((void**)&d_B, B_size, stream);
    cudaMallocAsync((void**)&d_C, C_size, stream);
    cudaMemcpyAsync(d_A, A, A_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, B_size, cudaMemcpyHostToDevice, stream);
    dim3 gridDim(CEIL_DIV(SIZE, BLOCK), CEIL_DIV(SIZE, BLOCK));
    dim3 blockDim(BLOCK, BLOCK);
    sharedMem<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, SIZE);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpyAsync(C, d_C, C_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    auto C_ref = new float[SIZE * SIZE];
    reference_gemm(SIZE, SIZE, SIZE, A, B, C_ref);
    // print(C, SIZE, SIZE);
    // print(C_ref, SIZE, SIZE);
    compare_matrices(C, C_ref, SIZE, SIZE);

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
