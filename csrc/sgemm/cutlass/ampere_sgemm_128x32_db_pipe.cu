#include "cute/arch/copy_sm80.hpp"
#include "cute/layout.hpp"
#include <cute/tensor.hpp>

namespace ampere_sgemm_128x32_db_pipe {

template <typename strideA, typename strideB, typename strideC,
          typename sALayout, typename sBLayout,
          typename ctaShape, typename copyPolicyA, typename copyPolicyB, typename mmaPolicy>
__global__ void ampere_sgemm_128x32_db_pipe(
    int m, int n, int k, float alpha, float beta,
    const float* __restrict__ A, strideA stride_A,
    const float* __restrict__ B, strideB stride_B,
    float* __restrict__ C, strideC stride_C,
    ctaShape cta_shape, sALayout sA_layout, sBLayout sB_layout,
    copyPolicyA copy_A, copyPolicyB copy_B, mmaPolicy tiled_mma
) {
    using namespace cute;

    // opt for dynamic smem size here due to 48kb limitation on static alloc
    extern __shared__ float smem_buffer[];
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k), stride_A));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k), stride_B));
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(m, n), stride_C));

    auto coord = make_coord(blockIdx.y, blockIdx.x, _);
    auto gA = local_tile(mA, cta_shape, coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_shape, coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_shape, coord, Step<_1, _1, X>{});
    auto sA = make_tensor(make_smem_ptr(&smem_buffer[0]), sA_layout);
    auto sB = make_tensor(make_smem_ptr(&smem_buffer[cosize_v<sALayout>]), sB_layout);

    auto tA = copy_A.get_thread_slice(threadIdx.x);
    auto tAgA = tA.partition_S(gA);
    auto tAsA = tA.partition_D(sA);
    
    auto tB = copy_B.get_thread_slice(threadIdx.x);
    auto tBgB = tB.partition_S(gB);
    auto tBsB = tB.partition_D(sB);

    uint gmem_tiles = size<2>(gA);
    uint smem_pipes = size<2>(sA);
    uint rmem_blocks = size<2>(tAsA);
    
    // prefetch for first (smem_pipes - 1) pipes
    for (uint i = 0; i < smem_pipes - 1; ++i) {
        copy(copy_A, tAgA(_,_,_,i), tAsA(_,_,_,i));
        copy(copy_B, tBgB(_,_,_,i), tBsB(_,_,_,i));
        --gmem_tiles;
    }
    cp_async_fence();
    
    // if (thread0()) {
        // print(sA); print("\n");
        // print(tAgA.layout()); print("\n");
        // print(tAsA.layout()); print("\n");
    // }
}

void nn(int m, int n, int k, float alpha,
        const float* A, int ldA, const float* B, int ldB, float beta, float* C,
        int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
    auto stride_A = make_stride(Int<1>{}, ldA);
    auto stride_B = make_stride(ldB, Int<1>{});
    auto stride_C = make_stride(Int<1>{}, ldC);

    // on SM86, this config. limits 1 CTA per SM--(((128 x 32 x 3) x 2) x 4) = 98304 bytes / CTA
    // TODO: experiment w/ a smaller K-tile or less pipes to increase occupancy?
    constexpr int n_pipes = 3;
    auto sA_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape), Int<n_pipes>{}));
    auto sB_layout = make_layout(
        make_shape(size<1>(cta_shape), size<2>(cta_shape), Int<n_pipes>{}),
        make_stride(size<2>(cta_shape), Int<1>{}, size<1>(cta_shape) * size<2>(cta_shape))
    );
    constexpr uint smem_size = (cosize_v<decltype(sA_layout)> + cosize_v<decltype(sB_layout)>) * sizeof(float);

    // at any given stage, tiled copy has 1 pipe of data in-flight
    // we use cp.async.cg instruction here b/c occupancy is 1 CTA / SM; no use for L1 caching (no space anyways, carveout is 98%)
    auto copy_A = make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{})),
        make_layout(make_shape(Int<4>{}, Int<1>{}))
    );
    auto copy_B = make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
        make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, Int<4>{}), LayoutRight{})
    );
    auto mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})),
        Tile<_128, _128, decltype(select<2>(cta_shape))>{}
    );

    auto kernel = ampere_sgemm_128x32_db_pipe<decltype(stride_A), decltype(stride_B), decltype(stride_C),
                                              decltype(sA_layout), decltype(sB_layout), decltype(cta_shape),
                                              decltype(copy_A), decltype(copy_B), decltype(mma)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 block_dim(size(mma));
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))), size(ceil_div(n, select<1>(cta_shape))));
    kernel<<<grid_dim, block_dim, smem_size, nullptr>>>(
        m, n, k, alpha, beta, A, stride_A, B, stride_B, C, stride_C, cta_shape, sA_layout, sB_layout, copy_A, copy_B, mma
    );
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