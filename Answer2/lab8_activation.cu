#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// ReLU Activation Kernel
// ReLU(x) = max(0, x)
// 這是典型的 Memory Bound 操作：運算簡單，瓶頸在記憶體讀寫
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

int main() {
    int M = 2048;
    int N = 2048;
    int size = M * N;
    
    std::cout << "資料大小: " << M << " x " << N << std::endl;

    // 配置 Managed Memory
    float *d_data;
    cudaMallocManaged(&d_data, size * sizeof(float));

    // 初始化數據 (模擬 GEMM + Bias 的輸出，有正有負)
    for(int i = 0; i < size; i++) {
        d_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    // 設定執行配置
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Warmup
    relu_kernel<<<blocks, threads>>>(d_data, size);
    cudaDeviceSynchronize();

    // 重新初始化
    for(int i = 0; i < size; i++) {
        d_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    // 計時開始
    std::cout << "開始執行 ReLU Activation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 執行 ReLU Kernel (多次執行以便觀察)
    int iterations = 1000;
    for(int i = 0; i < iterations; i++) {
        relu_kernel<<<blocks, threads>>>(d_data, size);
    }

    cudaDeviceSynchronize();

    // 計時結束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "ReLU 執行時間 (" << iterations << " 次): " << diff.count() << " s" << std::endl;
    std::cout << "平均每次: " << diff.count() / iterations * 1000 << " ms" << std::endl;

    // 驗證 ReLU 結果 (負數應變成 0)
    int positive_count = 0;
    int zero_count = 0;
    for(int i = 0; i < 100; i++) {  // 只檢查前 100 個
        if (d_data[i] > 0) positive_count++;
        if (d_data[i] == 0) zero_count++;
    }
    std::cout << "驗證 (前 100 個)：正數 = " << positive_count << ", 零 = " << zero_count << std::endl;

    cudaFree(d_data);
    return 0;
}
