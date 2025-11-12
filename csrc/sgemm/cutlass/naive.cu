#include <cute/tensor.hpp>

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
