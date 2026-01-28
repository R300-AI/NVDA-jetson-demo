#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

// ReLU Activation Kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，比較 Memory Bound vs Compute Bound" << std::endl;

    // ========== TODO 1: 承接 Practice 7 的結果矩陣 C' ==========
    int M = 2048;
    int N = 2048;
    int size = M * N;
    std::cout << "資料大小: " << M << " x " << N << std::endl;

    float *d_data;
    cudaMallocManaged(&d_data, size * sizeof(float));

    // ========== 初始化數據 (有正有負，用於測試 ReLU) ==========
    for (int i = 0; i < size; i++) {
        d_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    // ========== TODO 2: 執行 ReLU Kernel ==========
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    std::cout << "開始執行 ReLU Activation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        relu_kernel<<<blocks, threads>>>(d_data, size);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ========== 輸出結果 ==========
    std::cout << "ReLU 執行時間 (" << iterations << " 次): " << diff.count() << " s" << std::endl;
    std::cout << "平均每次: " << diff.count() / iterations * 1000 << " ms" << std::endl;

    // ========== 釋放記憶體 ==========
    cudaFree(d_data);
    return 0;
}
