#include <cute/tensor.hpp>

namespace ampere_sgemm_128x32_db_pipe {

template <typename strideA, typename strideB, typename strideC,
          typename sALayout, typename sBLayout,
          typename ctaShape, typename copyPolicy, typename copyPolicyB, typename mmaPolicy>
__global__ void ampere_sgemm_128x32_db_pipe(
    int m, int n, int k, float alpha, float beta,
    const float* __restrict__ A, strideA stride_A,
    const float* __restrict__ B, strideB stride_B,
    float* __restrict__ C, strideC stride_C,
    ctaShape cta_shape, sALayout sA_layout, sBLayout sB_layout,
    copyPolicy copy_A, copyPolicyB copy_B, mmaPolicy tiled_mma
) {

}

void nn(int m, int n, int k, float alpha,
              const float* A, int ldA, const float* B, int ldB, float beta, float* C,
              int ldC, cudaStream_t stream = 0) {
    
}

}  // namespace ampere_sgemm_128x32_db_pipe

void launch_ampere_sgemm_128x32_db_pipe(
    char transA, char transB, int m, int n, int k,
    float alpha, const float* A, int ldA,
    const float* B, int ldB, float beta,
    float* C, int ldC, cudaStream_t stream = 0
) {
    if (transA == 'N' && transB == 'N') {
        ampere_sgemm_128x32_db_pipe::nn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
}