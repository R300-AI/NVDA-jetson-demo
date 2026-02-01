#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

// ========== TODO: 實作 Divergent Kernel ==========
// 偶數/奇數 thread 執行不同操作 -> Warp 內分歧

__global__ void divergent_kernel(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    /* 請實作 Divergent Kernel：
       若 tid % 2 == 0（偶數），執行 C[tid] = A[tid] + B[tid]
       若 tid % 2 == 1（奇數），執行 C[tid] = A[tid] - B[tid]
    */
    if (tid % 2 == 0) {
        C[tid] = A[tid] + B[tid];
    } else {
        C[tid] = A[tid] - B[tid];
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile --capture-range=cudaProfilerApi 監測" << std::endl;

    // ========== TODO 1: 設定向量大小與配置記憶體 ==========
    int N = 10000000;               /* 請將向量大小改為 10000000 (10^7) */
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    /* 請使用 cudaMallocManaged 配置 B, C 的記憶體 */
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // ========== 初始化向量數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A, N);
    fill_random_uniform_0_1(B, N);

    // ========== TODO 2: GPU 執行配置 ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;   /* 請計算正確的 Block 數量: (N + threads - 1) / threads */

    // ========== 開始 Profiling（僅追蹤 GPU Kernel）==========
    cudaProfilerStart();

    // ========== 測試 Divergent Kernel ==========
    auto start = std::chrono::high_resolution_clock::now();
    divergent_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 ==========
    std::cout << "Divergent Kernel Time: " << elapsed.count() << " s" << std::endl;

    // ========== 停止 Profiling ==========
    cudaProfilerStop();

    // ========== 釋放記憶體 ==========
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
