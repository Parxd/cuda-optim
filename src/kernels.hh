#ifndef KERNELS_HH
#define KERNELS_HH

#include "kernel/0_naive.hh"
#include "kernel/1_smem.hh"
#include "kernel/2_1dim_threadtile.hh"
#include "kernel/3_2dim_threadtile.hh"
#include "kernel/4_vectorize.hh"
#include "kernel/5_warptile.hh"
#include "kernel/x_cublas.hh"

inline void test_kernel(int kernel, int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    switch (kernel) {
    case 0:
        launch_naive(M, N, K, A, B, C, stream);
        break;
    case 1:
        launch_smem(M, N, K, A, B, C, stream);
        break;
    case 2:
        launch_onedim_threadtile(M, N, K, A, B, C, stream);
        break;
    case 3:
        launch_twodim_threadtile(M, N, K, A, B, C, stream);
        break;
    case 4:
        launch_vectorize(M, N, K, A, B, C, stream);
        break;
    case 5:
        launch_warptile(M, N, K, A, B, C, stream);
        break;
    case 10:
        launch_cublas(M, N, K, A, B, C, stream);
        break;
    default:
        break;
    }
}

#endif