#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <int BM, int BN, int BK>
__device__ void ld_global2shared(
    const cg::thread_block& cta, int M, int N, int K, float* gA, float* gB, float* sA, float* sB
) {
    constexpr int thr_id = cta.thread_rank();
    constexpr int cta_id = cta.group_index().x;
    constexpr int cta_row = cta_id / (M / BM);
    constexpr int cta_col = cta_id % (N / BK);

    constexpr int A_lds = (BM * BK / 4) / cta.num_threads();
    constexpr int A_ld_rows = BM / A_lds;
    constexpr int B_lds = (BK * BN / 4) / cta.num_threads();
    constexpr int B_ld_rows = BK / B_lds;
    // mapping layouts (NUM_THREADS_IN_CTA, 1) --> (BM, BK / 4)
    constexpr int A_row = thr_id / (BK / 4);
    constexpr int A_col = thr_id % (BK / 4);
    // mapping layouts (NUM_THREADS_IN_CTA, 1) --> (BK, BN / 4);
    constexpr int B_row = thr_id / (BN / 4);
    constexpr int B_col = thr_id % (BN / 4);
    
    const int tiles = K / BK;
    for (int tile = 0; tile < tiles; ++tile) {
        for (ld = 0; ld < A_lds; ++ld) {
            float4 vector = reinterpret_cast<float4*>(
                &gA[(cta_row * BM + (ld * A_ld_rows + A_row)) * K + (tile * BK + (A_col * 4))]
            )[0];
            sA[(A_col * 4    ) * BM + (A_ld_rows * ld + A_row)] = vector.x;
            sA[(A_col * 4 + 1) * BM + (A_ld_rows * ld + A_row)] = vector.y;
            sA[(A_col * 4 + 2) * BM + (A_ld_rows * ld + A_row)] = vector.z;
            sA[(A_col * 4 + 3) * BM + (A_ld_rows * ld + A_row)] = vector.w;
        } 
        for (ld = 0; ld < B_lds; ++ld) {
            reinterpret_cast<float4*>(&sB[(ld * B_ld_rows + B_row) * BN + (B_col * 4)])[0] = 
                reinterpret_cast<float4*>(
                    &gB[(tile * BK + (ld * B_ld_rows + B_row)) * N + (cta_col * BN + (B_col * 4))]
                )[0];
        }
    }
    cta.sync();
}

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int WIM, int WIN,
          int TM, int TN, int TK>
__global__ void warptile(int M, int N, int K, float* A, float* B, float* C) {
    __shared__ float A_tile[BM * BK];
    __shared__ float B_tile[BK * BN];
    float A_register[TM * TK * WIM] = {0.0};
    float B_register[TK * TN * WIN] = {0.0};
    float res[TM * TN * WIM * WIN] = {0.0};

    cg::thread_block cta = cg::this_thread_block();
    ld_global2shared<BM, BN, BK>(cta, M, N, K, A, B, A_tile, B_tile);
    
    auto warp = cg::thread_block_tile<32>(cta);
}

__host__ void launch_warptile(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    // CTA size
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    // warp-tile size
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 1;
    // thread-tile size
    constexpr int TM = 16;
    constexpr int TN = 8;
    constexpr int TK = 1;
    // iters. over warp-tile
    // assume arbitrary (4, 8) thread layout in warp
    constexpr int WIM = TM * 4 / WM;
    constexpr int WIN = TN * 8 / WN;

    constexpr int threads_per_warp = 32;
    constexpr int threads_per_cta = (BM * BN) / (WM * WN) * threads_per_warp;

    static_assert(WK >= TK);
    // TODO: need more static asserts here
    dim3 block_dim(threads_per_cta);
    dim3 grid_dim(CEIL_DIV(M, BM) * CEIL_DIV(N, BN));
    warptile<BM, BN, BK, WM, WN, WK, WIM, WIN, TM, TN, TK><<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, B, C);
}