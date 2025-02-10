#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <int BM, int BN, int BK>
__device__ void global2shared(
    cg::thread_block cta, int M, int N, int K, float* gA, float* gB, float* sA, float* sB
) {
    dim3 block_idx = cta.group_index();
    
    cta.sync();
}


template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int WIM, int WIN,
          int TM, int TN, int TK>
__global__ void warptile(int M, int N, int K, float* A, float* B, float* C) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float A_tile[BM * BK];
    __shared__ float B_tile[BK * BN];
    
    float A_register[TM * TK * WIM] = {0.0};
    float B_register[TK * TN * WIN] = {0.0};
    float res[TM * TN * WIM * WIN] = {0.0};
    global2shared<BM, BN, BK>(cta, M, N, K, A, B, A_tile, B_tile);
    
    auto warp = cg::thread_block_tile<32>(cta);

    const int tiles = CEIL_DIV(K, BK);
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
    // warp-tile iterations
    constexpr int WIM = 2;
    constexpr int WIN = 2;
    // thread-tile size
    constexpr int TM = 16;
    constexpr int TN = 8;
    constexpr int TK = 8;
    
    constexpr int threads_per_warp = 32;
    constexpr int elements_per_block = BM * BK;
    constexpr int elements_per_warp = WM * WK;
    constexpr int elements_per_thread = TM * TN;
    
    dim3 block_dim{};
    dim3 grid_dim(CEIL_DIV(M, BM) * CEIL_DIV(N, BN));
    warptile<BM, BN, BK, WM, WN, WK, WIM, WIN, TM, TN, TK><<<grid_dim, block_dim, 0, stream>>>(M, N, K, A, B, C);
}