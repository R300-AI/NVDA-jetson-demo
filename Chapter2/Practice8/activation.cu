#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// ========== TODO: 實作 ReLU Activation Kernel ==========
// ReLU(x) = max(0, x)
// 這是典型的 Memory Bound 操作：運算簡單，瓶頸在記憶體讀寫
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        /* 請實作 ReLU: data[idx] = max(0, data[idx]) */
        /* 提示: 使用 fmaxf(0.0f, data[idx]) */
        
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，比較 Memory Bound vs Compute Bound" << std::endl;

    // ========== TODO 1: 承接 Practice 7 的結果矩陣 C' ==========
    // (或重新建立 2048×2048 矩陣模擬)

    int M = 1024;                   /* 請填入與 Practice7 相同的大小 (2048) */
    int N = 1024;
    int size = M * N;
    
    std::cout << "資料大小: " << M << " x " << N << std::endl;

    float *d_data;
    /* 請配置 Managed Memory */


    // ========== 初始化數據 (模擬 GEMM + Bias 的輸出) ==========
    // 有正有負的數值，用於測試 ReLU 效果
    for(int i = 0; i < size; i++) {
        d_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }


    // ========== TODO 2: 實作 ReLU Kernel，計算 R = max(0, C') ==========

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Warmup
    relu_kernel<<<blocks, threads>>>(d_data, size);
    cudaDeviceSynchronize();


    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    std::cout << "開始執行 ReLU Activation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 因為單次執行太快，執行多次以便觀察
    int iterations = 1000;
    for(int i = 0; i < iterations; i++) {
        /* 請執行 relu_kernel */
        
    }

    cudaDeviceSynchronize();


    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    // ========== 輸出結果 ==========
    std::cout << "ReLU 執行時間 (" << iterations << " 次): " << diff.count() << " s" << std::endl;
    std::cout << "平均每次: " << diff.count() / iterations * 1000 << " ms" << std::endl;


    // ========== TODO 4: 驗證 ReLU 結果 ==========
    // 檢查負數是否都變成 0

    /* 可選：印出部分結果驗證正確性 */


    // ========== 清理資源 ==========
    cudaFree(d_data);
    
    return 0;
}
