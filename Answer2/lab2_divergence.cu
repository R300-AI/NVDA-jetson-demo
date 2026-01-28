#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

// Divergent Kernel: 偶數/奇數 thread 執行不同操作
__global__ void divergent_kernel(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (tid % 2 == 0) {
            C[tid] = A[tid] + B[tid];  // 偶數: 加法
        } else {
            C[tid] = A[tid] - B[tid];  // 奇數: 減法
        }
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 Warp Stall Reasons" << std::endl;

    // ========== TODO 1: 設定向量大小與配置記憶體 ==========
    int N = 10000000;  // 10^7
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // ========== 初始化向量數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A, N);
    fill_random_uniform_0_1(B, N);

    // ========== TODO 2: GPU 執行配置 ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ========== 測試 Divergent Kernel ==========
    auto start = std::chrono::high_resolution_clock::now();
    divergent_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 ==========
    std::cout << "Divergent Kernel Time: " << elapsed.count() << " s" << std::endl;

    // ========== 釋放記憶體 ==========
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
