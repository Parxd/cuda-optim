#include <iostream>
#include <cuda_runtime.h>
#include "../utils.h"

void run_kernel(int);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number>\n";
        return 1;
    }
    int kernel = std::atoi(argv[1]);

    int m, n, k;
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C(m * n);
    fill_random(h_A.data(), h_A.size(), 0.0, 1.0);
    fill_random(h_B.data(), h_B.size(), 0.0, 1.0);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    run_kernel(kernel);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));



    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}