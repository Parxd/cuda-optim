#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include <cute/tensor.hpp>

template <typename ctaShape, typename sALayout, typename sBLayout, typename sCLayout,
          typename copyPolicy, typename mmaPolicy>
__global__ void gemm_kernel(
    int m, int n, int k, float alpha, float beta,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    ctaShape cta_shape, sALayout sA_layout, sBLayout sB_layout, sCLayout sC_layout,
    copyPolicy tiled_copy, mmaPolicy tiled_mma
) {
    using namespace cute;
    // TODO: ADD ASSERTIONS
    __shared__ float sA_buffer[cosize_v<sALayout>];
    __shared__ float sB_buffer[cosize_v<sBLayout>];
    auto blk_coord = make_coord(blockIdx.y, blockIdx.x, _);

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(m, k), LayoutLeft{});
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(n, k), LayoutLeft{});
    auto gA = local_tile(mA, cta_shape, blk_coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_shape, blk_coord, Step<X, _1, _1>{});
    auto sA = make_tensor(make_smem_ptr(sA_buffer), sA_layout);
    auto sB = make_tensor(make_smem_ptr(sB_buffer), sB_layout);

    auto tA = tiled_copy.get_thread_slice(threadIdx.x);
    auto tAgA = tA.partition_S(gA);
    auto tAsA = tA.partition_D(sA);
    auto tArA = make_fragment_like(tAsA);

    /*
    gmem_ptr[32b](0x7daf51e00000) o ((_4,_1),_1,_1,1):((_1,_0),_0,_0,1024)
    ptr[32b](0x7daf5ffffcd0) o ((_4,_1),_1,_1):((_1,_0),_0,_0)
    CUTE_GEMM:     [   0.1]GFlop/s  (1.8265)ms
    */

    if (thread0()) {
        // copy(tiled_copy, tAgA(_,_,0,0), tArA(_,_,0));
        print("  tAgA: "); print(tAgA); print("\n");
        print("  tArA: "); print(tArA); print("\n");
        // print(tAgA(_,_,0,_)); print("\n");
        print(dice(Step<_1,_1,X,_1>{}, tAgA.shape()));
    }

    if(thread0()) {
        // print(blk_coord); print("\n");
        // print("  gA: "); print(gA); print("\n");
        // print("  tAgA: "); print(tAgA); print("\n");
        // print("  tAsA: "); print(tAsA); print("\n");
        // print("  tArA: "); print(tArA); print("\n");
    }
}

void sgemm_nn(int m, int n, int k, float alpha,
              const float* A, int ldA, const float* B, int ldB, float beta, float* C,
              int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    auto problem_shape = make_shape(m, n, k);  // dynamic
    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto sA_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape)), LayoutLeft{});
    auto sB_layout = make_layout(make_shape(select<1>(cta_shape), select<2>(cta_shape)), LayoutRight{});
    auto sC_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape)), LayoutLeft{});
    // print(cosize_v<decltype(sA_layout)>);

    TiledCopy copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), LayoutLeft{}),
        make_layout(make_shape(Int<4>{}, Int<1>{}), LayoutLeft{})
    );
    TiledMMA mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{},
        make_layout(make_shape(Int<2>{}, Int<2>{}))
    );
    
    dim3 cta_dim(size(mma));
    dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))),
                  size(ceil_div(n, select<1>(cta_shape))));
    gemm_kernel<<<grid_dim, cta_dim, 0, stream>>>(
        m, n, k, alpha, beta, A, B, C, cta_shape, sA_layout, sB_layout, sC_layout, copy, mma
    );
}

void launch_pipeline_nodb_g2r(
    char transA, char transB, int m, int n, int k,
    float alpha, const float* A, int ldA,
    const float* B, int ldB, float beta,
    float* C, int ldC, cudaStream_t stream = 0
) {
    if (transA == 'N' && transB == 'N') {
        sgemm_nn(m, n, k, alpha, A, m, B, k, beta, C, m);
    }
}