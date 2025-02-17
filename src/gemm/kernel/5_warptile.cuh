 #include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <int BM, int BN, int BK,
          int WM, int WN>
__device__ void ld_global2shared(
    const cg::thread_block& cta, int tile, int M, int N, int K, float* gA, float* gB, float* sA, float* sB
) {
    const int thr_id = cta.thread_rank();
    const int cta_id = cta.group_index().x;
    const int cta_row = cta_id / (N / BN);
    const int cta_col = cta_id % (N / BN);
    
    // avoid cta.num_threads() for constexpr
    constexpr int cta_size = (BM * BN) / (WM * WN) * 32;
    constexpr int A_lds = (BM * BK / 4) / cta_size;
    constexpr int A_ld_rows = BM / A_lds;
    constexpr int B_lds = (BK * BN / 4) / cta_size;
    constexpr int B_ld_rows = BK / B_lds;
    // mapping layouts (NUM_THREADS_IN_CTA, 1) -> (BM, BK / 4)
    const int A_row = thr_id / (BK / 4);
    const int A_col = thr_id % (BK / 4);
    // mapping layouts (NUM_THREADS_IN_CTA, 1) -> (BK, BN / 4);
    const int B_row = thr_id / (BN / 4);
    const int B_col = thr_id % (BN / 4);
    
    for (int ld = 0; ld < A_lds; ++ld) {
        auto [x, y, z, w] = reinterpret_cast<float4*>(
            &gA[(cta_row * BM + (ld * A_ld_rows + A_row)) * K + (tile * BK + (A_col * 4))]
        )[0];
        sA[(A_col * 4    ) * BM + (A_ld_rows * ld + A_row)] = x;
        sA[(A_col * 4 + 1) * BM + (A_ld_rows * ld + A_row)] = y;
        sA[(A_col * 4 + 2) * BM + (A_ld_rows * ld + A_row)] = z;
        sA[(A_col * 4 + 3) * BM + (A_ld_rows * ld + A_row)] = w;
    }
    for (int ld = 0; ld < B_lds; ++ld) {
        reinterpret_cast<float4*>(&sB[(ld * B_ld_rows + B_row) * BN + (B_col * 4)])[0] = 
            reinterpret_cast<float4*>(
                &gB[(tile * BK + (ld * B_ld_rows + B_row)) * N + (cta_col * BN + (B_col * 4))]
            )[0];
    }
}

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int WIM, int WIN,
          int TM, int TN, int TK>
__device__ void ld_shared2reg_mma(
    const cg::thread_block_tile<32, cg::thread_block>& warp, int warp_k, float* sA, float* sB, float* rA, float* rB
) {
    // tile CTA-level C with (warp_rows, warp_cols) layout of warps
    // assert(warp_rows * warp_cols == warp.meta_group_size())
    constexpr int warp_rows = BM / WM;
    constexpr int warp_cols = BN / WN;
    // which warp are we in?
    const int warp_m = warp.meta_group_rank() / warp_cols;
    const int warp_n = warp.meta_group_rank() % warp_cols;
    // ld sA -> rA
    for (int warp_iter_m = 0; warp_iter_m < WIM; ++warp_iter_m) {
        for (int thread_m = 0; thread_m < TM; ++thread_m) {
            rA[warp_iter_m * TM + thread_m] = 
                sA[warp_k * BM + (warp_m * WM + (warp_iter_m * WIM + thread_m))];
        }
    }
    // ld sB -> rB
    for (int warp_iter_n = 0; warp_iter_n < WIN; ++warp_iter_n) {
        for (int thread_n = 0; thread_n < TN; ++thread_n) {
            rB[warp_iter_n * TN + thread_n] = 
                sB[warp_k * BN + (warp_n * WN + (warp_iter_n * WIN + thread_n))];
        }
    }
    // mma
    for (int warp_iter_n = 0; warp_iter_n < WIN; ++warp_iter_n) {
        for (int warp_iter_m = 0; warp_iter_m < WIM; ++warp_iter_m) {
            for (int thread_n = 0; thread_n < TN; ++thread_n) {
                for (int thread_m = 0; thread_m < TM; ++thread_m) {
                    
                }
            }
        }
    }
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
    
    const int tiles = K / BK;
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32, cg::thread_block>(cta);
    for (int tile = 0; tile < tiles; ++tile) {
        ld_global2shared<BM, BN, BK, WM, WN>(cta, tile, M, N, K, A, B, A_tile, B_tile);
        cta.sync();
        
        for (int warp_k = 0; warp_k < BK; ++warp_k) {
            ld_shared2reg_mma<BM, BN, BK, WM, WN, WK, WIM, WIN, TM, TN, TK>(
                warp, warp_k, A_tile, B_tile, A_register, B_register
            );
        }
    }
}

__host__ inline void launch_warptile(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    // CTA size
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    // warp-tile size
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WK = 1;
    // thread-tile size
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int TK = 1;
    // iters. over warp-tile
    // assume arbitrary (4, 8) thread layout in warp
    constexpr int WIM = WM / (TM * 8);
    constexpr int WIN = WN / (TN * 4);
    
    constexpr int threads_per_cta = (BM * BN) / (WM * WN) * 32;
    
    static_assert(WK >= TK);
    // TODO: need more static asserts here
    dim3 block_dim(threads_per_cta);
    dim3 grid_dim((M / BM) * (N / BN));
    warptile<BM, BN, BK, WM, WN, WK, WIM, WIN, TM, TN, TK>
        <<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, B, C);
}