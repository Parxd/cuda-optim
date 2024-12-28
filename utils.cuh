#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>

#define IDX2C(i, j, stride) (i * stride + j)  // i = threadIdx.y, j = threadIdx.x
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define THREAD_WARP_ID(idx, idy, block_x_dim) ((idx + idy * block_x_dim) / 32)
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA Error in " << __FILE__               \
                      << " at line " << __LINE__ << ": "            \
                      << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

void fill_zeros(float* ptr, int size) {
    std::fill(ptr, ptr + size, float(0.0));
}

void fill_ones(float* ptr, int size) {
    std::fill(ptr, ptr + size, float(1.0));
}

void fill_increment(float* ptr, int size) {
    std::iota(ptr, ptr + size, 1);
}

void fill_random(float* arr, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5, 1.0);
    std::generate(arr, arr + size, [&]() { return dis(gen); });
}

void print(float* ptr, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << ptr[IDX2C(i, j, c)] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void cudaDeviceInfo() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) 
                  << ": " << cudaGetErrorString(error_id) << std::endl;
        return;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s):" << std::endl;
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max grid size: [" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Max threads dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Multi-processors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
    }
}

void reference_gemm(int M, int N, int K, float* A, float* B, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0;
            for (int k = 0; k < K; k++) {
                tmp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
}

bool compare_matrices(float* A, float* B, int rows, int cols, float tol=0.000001) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = std::fabs(A[i * cols + j] - B[i * cols + j]);
            if (diff > tol) {
                printf("Mismatch at (%d, %d): A=%.6f, B=%.6f, Diff=%.6f\n", 
                       i, j, A[i * cols + j], B[i * cols + j], diff);
                return false;
            }
        }
    }
    printf("Success!\n");
    return true;
}
