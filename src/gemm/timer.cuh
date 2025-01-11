#include "kernels.cuh"
#include "../utils.cuh"

int time_kernel(int kernel_num,
         int M, int N, int K,
         float* A, float* B, float* C,
         cudaStream_t stream,
         int trials) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    test_kernel(kernel_num, M, N, K, A, B, C, stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel time: (%f) ms.\n", ms);
    // printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
            // time, 2. * 1e-9 * trials * M * N * K / elapsed_time, M);
    fflush(stdout);
    return 0;
}