#include "kernels.cuh"
#include "../utils.cuh"

int time_kernel(int kernel_num,
         int M, int N, int K,
         float* A, float* B, float* C,
         cudaStream_t stream,
         int trials) {
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    cudaEventRecord(beg);
    for (int i = 0; i < trials; i++) {
        test_kernel(kernel_num, M, N, K, A, B, C, stream);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0;

    printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
            elapsed_time / trials, 2. * 1e-9 * trials * M * N * K / elapsed_time, M);
    fflush(stdout);
    return 0;
}