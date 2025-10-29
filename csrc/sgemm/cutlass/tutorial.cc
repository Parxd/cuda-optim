#include "cute/arch/mma_sm70.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print_latex.hpp"
#include "cute/util/print_tensor.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include <assert.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <class TA, class TB, class TC, class Alpha, class Beta, class sALayout,
          class sBLayout>
__global__ void gemm_kernel(int m, int n, int k, Alpha alpha, Beta beta,
                            const TA *A, const TB *B, TC *C, sALayout sA,
                            sBLayout sB) {
    using namespace cute;
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tt(char transA, char transB, int m, int n, int k, Alpha alpha,
             TA const *A, int ldA, TB const *B, int ldB, Beta beta, TC *C,
             int ldC, cudaStream_t stream = 0) {

    using namespace cute;

    auto problem_shape = make_shape(m, n, k); // runtime
    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto sA = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape)),
                        LayoutRight{});
    auto sB = make_layout(make_shape(select<1>(cta_shape), select<2>(cta_shape)),
                        LayoutLeft{});
    // auto sC = make_layout(

    // );
    auto thrA = make_layout(make_shape());

    dim3 cta_dim(1);
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))),
                size(ceil_div(n, select<1>(cta_shape))));
    gemm_kernel<<<grid_dim, cta_dim, 0, stream>>>(m, n, k, alpha, beta, A, B, C,
                                                sA, sB);
}

/*
uses (1:1) TV copy & compute mapping 
*/
template <class AStride, class BStride, class CStride,
          class ThreadLayout>
__global__ void elementwise_kernel_v1(
    int M, int N, float* A, AStride dA, float* B, BStride dB, float* C, CStride dC,
   ThreadLayout thrLayout) {

    using namespace cute;
    
    auto mA = make_tensor(A, make_shape(M, N), dA);
    auto mB = make_tensor(B, make_shape(M, N), dB);
    auto mC = make_tensor(C, make_shape(M, N), dC);
    auto gA = local_tile(mA, thrLayout.shape(), make_coord(blockIdx.x, blockIdx.y));  // CTA-local views
    auto gB = local_tile(mB, thrLayout.shape(), make_coord(blockIdx.x, blockIdx.y));
    auto gC = local_tile(mC, thrLayout.shape(), make_coord(blockIdx.x, blockIdx.y));
    auto tgA = local_partition(gA, thrLayout, threadIdx.x);
    auto tgB = local_partition(gB, thrLayout, threadIdx.x);
    auto tgC = local_partition(gC, thrLayout, threadIdx.x);
    auto tgC_reg = make_tensor_like(tgC);
    for (uint i = 0; i < size(tgA); ++i) {
        tgC_reg(i) = tgA(i) + tgB(i);
    }
    copy(tgC_reg, tgC);

    #if 1
    if (thread(1, 0)) {
        print("  mA : "); print(mA); print("\n");
        print("  gA : "); print(gA); print("\n");
        print("  tgA : "); print(tgA); print("\n");
        print("  tgC : "); print_tensor(tgC); print("\n");
        print("  tgC_res : "); print_tensor(tgC_reg); print("\n");
        print("  gC : "); print_tensor(gC); print("\n");
    }
    #endif
}

int main(int argc, char **argv) {
    // cute::device_init(0);

    // int m = 32;
    // int n = 64;
    // int k = 16;
    // char transA = 'T';
    // char transB = 'T';
    // if (transA == 'N' || transB == 'N') {
    //     assert(false);  // only TT sgemm for now
    // }

    // using TA = float;
    // using TB = float;
    // using TC = float;
    // using TI = float;
    // TI alpha = 1.0;
    // TI beta  = 0.0;
    // std::cout << "M = " << m << std::endl;
    // std::cout << "N = " << n << std::endl;
    // std::cout << "K = " << k << std::endl;

    // thrust::host_vector<TA> h_A(m*k);
    // thrust::host_vector<TB> h_B(n*k);
    // thrust::host_vector<TC> h_C(m*n);

    // for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(2 * (rand() /
    // double(RAND_MAX)) - 1); for (int j = 0; j < n * k; ++j) h_B[j] =
    // static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1); for (int j = 0; j < m
    // * n; ++j) h_C[j] = static_cast<TC>(-1);

    // thrust::device_vector<TA> d_A = h_A;
    // thrust::device_vector<TB> d_B = h_B;
    // thrust::device_vector<TC> d_C = h_C;

    // int ldA = 0, ldB = 0, ldC = m;
    // d_C = h_C;
    // gemm_tt(transA, transB, m, n, k, alpha, d_A.data().get(), ldA,
    // d_B.data().get(), ldB, beta, d_C.data().get(), ldC); CUTE_CHECK_LAST();
    // thrust::host_vector<TC> cute_result = d_C;

    // double gflops = (2.0 * m * n * k) * 1e-9;
    // const int timing_iterations = 10;
    // GPU_Clock timer;
    // timer.start();
    // for (int i = 0; i < timing_iterations; ++i) {
    //     gemm_tt(transA, transB, m, n, k,
    //     alpha,
    //     d_A.data().get(), ldA,
    //     d_B.data().get(), ldB,
    //     beta,
    //     d_C.data().get(), ldC);
    // }
    // double cute_time = timer.seconds() / timing_iterations;
    // CUTE_CHECK_LAST();
    // printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time,
    // cute_time*1000);

    // ====================================
    // element-wise op. kernel (T: M-major)
    // ====================================
    using namespace cute;
    int M = 8; int N = 64;
    auto hA = thrust::host_vector<float>(M * N);
    auto hB = thrust::host_vector<float>(M * N);
    auto hC = thrust::host_vector<float>(M * N);
    std::iota(hA.begin(), hA.end(), 0.0f);
    std::iota(hB.begin(), hB.end(), 0.0f);
    
    auto dA = thrust::device_vector<float>(M * N);
    auto dB = thrust::device_vector<float>(M * N);
    auto dC = thrust::device_vector<float>(M * N, 0);
    dA = hA; dB = hB;

    auto str_A = make_stride(N, Int<1>{});
    auto str_B = make_stride(N, Int<1>{});
    auto str_C = make_stride(N, Int<1>{});
    auto thrLayout = make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto valLayout = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    // auto ctaShape = product_each(raked_product(thrLayout, valLayout).shape());  // for 128-bit copy kernel

    dim3 blockDim(size(thrLayout));
    dim3 gridDim(M / size<0>(thrLayout), N / size<1>(thrLayout));
    print("blockDim: "); print(blockDim); print("\n");
    print("gridDim: "); print(gridDim); print("\n");
    elementwise_kernel_v1<<<gridDim, blockDim, 0, nullptr>>>(
        M, N, dA.data().get(), str_A, dB.data().get(), str_B, dC.data().get(), str_C, thrLayout
    );

    return 0;
}
