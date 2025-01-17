#include "kernels.cuh"
#include "../utils.cuh"
#include "../arg_parser.h"
#include "timer.cuh"

int main(int argc, char* argv[]) {
    int kernel_id = 0;
    int M = 512;
    int N = 512;
    int K = 512;
    bool time_flag = false;
    bool verify_flag = true;
    
    args::ArgumentParser parser("CUDA SGEMM Benchmarking");
    args::HelpFlag help(parser, "help", "display this help menu", {'h', "help"});
    args::ValueFlag<int> kernel(parser, "kernel number", "kernel to launch (int)", {'k'});
    args::ValueFlag<bool> time(parser, "time", "time kernel performance (bool)", {'t'});
    args::ValueFlag<bool> verify(parser, "verify", "verify kernel correctness (bool)", {'v'});
    args::PositionalList<int> sizes(parser, "sizes", "problem size tuple (M, N, K)");
    try {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    if (kernel) { kernel_id = args::get(kernel); }
    if (time) { time_flag = bool(args::get(time)); }
    if (verify) { verify_flag = bool(args::get(verify)); }
    if (sizes) {
        auto res = args::get(sizes);
        M = res[0];
        N = res[1];
        K = res[2];
    }
    std::cout << "Benchmarking CUDA SGEMM...\n" << 
        "- Kernel: " << kernel_id << "\n" <<
        "- Timing: " << time_flag << "\n" <<
        "- Verify: " << verify_flag << "\n" <<
        "- M = " << M << "\n" <<
        "- N = " << N << "\n" <<
        "- K = " << K << "\n";

    // ------------------------------------------------------------------------------------------
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float *d_A, *d_B, *d_C;
    auto A = new float[M * K];
    auto B = new float[K * N];
    // fill_random(A, M * K, 0.1, 0.3);
    // fill_random(B, K * N, 0.1, 0.3);
    fill_increment(A, M * K);
    fill_increment(B, K * N);
    
    auto C = new float[M * N];
    auto A_size = sizeof(float) * M * K;
    auto B_size = sizeof(float) * K * N;
    auto C_size = sizeof(float) * M * N;
    cudaMallocAsync((void**)&d_A, A_size, stream);
    cudaMallocAsync((void**)&d_B, B_size, stream);
    cudaMallocAsync((void**)&d_C, C_size, stream);
    cudaMemcpyAsync(d_A, A, A_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, B_size, cudaMemcpyHostToDevice, stream);
    
    test_kernel(kernel_id, M, N, K, d_A, d_B, d_C, stream);
    cudaMemcpyAsync(C, d_C, C_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (verify_flag) {
        auto C_ref = new float[M * N];
        reference_gemm(M, N, K, A, B, C_ref);
        compare_matrices(C, C_ref, M, N, 0.001);    
    }
    if (time_flag) {
        time_kernel(kernel_id, M, N, K, d_A, d_B, d_C, stream, 10);
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    cudaStreamDestroy(stream);
    return 0;
}
