template <int BM, int BN, int BK, int WM, int WN, int WIM, int WIN, int TM, int TN, int threads>
__global__ void __launch_bounds__(threads) warptile_v2(int M, int N, int K, float* A, float* B, float* C) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    float rC[WIM * TM * WIN * TN] = {0.0};
    float rA[WIM * TM] = {0.0};
    float rB[WIN * TN] = {0.0};

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (threads * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = threads / (BN / 4);

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint warpIdx = threadIdx.x / 32;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    for (int tile = 0; tile < (K / BK); ++tile) {
        // load
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            // const float4 tmp = reinterpret_cast<const float4 *>(&A[
            //     (blockIdx.y * BM + (innerRowA + offset)) * K + (tile * BK + (innerColA * 4))]
            // )[0];
            const float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
            sA[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            sA[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            sA[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            sA[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            // reinterpret_cast<float4 *>(&sB[(innerRowB + offset) * BN + innerColB * 4])[0] =
            //     reinterpret_cast<const float4 *>(&B[tile * BK + (innerRowB + offset) * N + (blockIdx.x * BN + (innerColB * 4))])[0];
            reinterpret_cast<float4 *>(&sB[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();
    }
}

__host__ inline void launch_warptile_v2(int M, int N, int K, float* A, float* B, float* C, cudaStream_t stream) {
    int test = 50;
    int test2 = 50
}