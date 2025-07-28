#include <cublas_v2.h>

// final boss
void inline launch_cublas(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm_v2(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha, A, K, B, N, &beta, C, N
    );
}