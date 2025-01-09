#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "kernel/0_naive.cuh"
#include "kernel/1_shared_mem.cuh"
#include "kernel/2_onedim_blocktile.cuh"
#include "kernel/3_twodim_blocktile.cuh"
#include "kernel/x_cublas.cuh"

void test_kernel(int kernel, int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    switch (kernel) {
    case 0:
        launch_naive(M, N, K, A, B, C, stream);
        break;
    case 1:
        launch_shared_mem(M, N, K, A, B, C, stream);
        break;
    case 2:
        launch_onedim_blocktile(M, N, K, A, B, C, stream);
        break;
    case 3:
        launch_twodim_blocktile(M, N, K, A, B, C, stream);
        break;
    case 10:
        launch_cublas(M, N, K, A, B, C, stream);
        break;
    default:
        break;
    }
}

#endif