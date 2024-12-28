#include <iostream>
#include <assert.h>
#include "../../../utils.cuh"

#define SIZE 128

__global__ void naive(float* A, float* B, float* C, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < size && idy < size) {
        float tmp = 0.0f;
        for (int i = 0; i < size; ++i) {
            tmp += A[idy * size + i] * B[i * size + idx];
        }
        C[idy * size + idx] = tmp;
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
    dim3 gridDim(CEIL_DIV(SIZE, 32), CEIL_DIV(SIZE, 32));
    dim3 blockDim(32, 32);
    naive<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, SIZE);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpyAsync(C, d_C, C_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    auto C_ref = new float[SIZE * SIZE];
    reference_gemm(SIZE, SIZE, SIZE, A, B, C_ref);
    compare_matrices(C, C_ref, SIZE, SIZE, SIZE);
    
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
