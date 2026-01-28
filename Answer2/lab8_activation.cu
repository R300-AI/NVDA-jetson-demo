/*
 * Practice 8: Activation Functions 的處理效能
 * 
 * 題目要求：
 * 1. 使用上一題的結果矩陣 C' (這裡重新模擬)
 * 2. 在 GPU 上對 C' 進行 ReLU 操作，得到結果矩陣 R
 *    R = ReLU(C') = max(0, C + b)
 * 3. 利用 std::chrono 記錄整體執行時間
 * 
 * 編譯: nvcc lab8_activation.cu -o lab8 -O2 -arch=sm_87
 * 執行: ./lab8
 * 監測: nsys profile -o report_lab8 ./lab8
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

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
    const int N = 2048;
    const int size = N * N;
    
    std::cout << "=== Practice 8: ReLU Activation ===" << std::endl;
    std::cout << "資料大小: " << N << " x " << N << std::endl;

    // 配置 Managed Memory
    float* data;
    cudaMallocManaged(&data, size * sizeof(float));

    // 模擬 GEMM + Bias 的輸出 (有正有負)
    // 在真實情況中，這是 Practice 7 的輸出
    for (int i = 0; i < size; i++) {
        data[i] = (i % 3 == 0) ? -0.5f : 0.5f;  // 1/3 為負數
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Warmup
    relu_kernel<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();

    // 重新初始化 (因為 warmup 會改變資料)
    for (int i = 0; i < size; i++) {
        data[i] = (i % 3 == 0) ? -0.5f : 0.5f;
    }

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();

    relu_kernel<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // ========== 輸出結果 ==========
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "ReLU 執行時間: " << elapsed * 1000 << " ms" << std::endl;

    // 驗證 ReLU 結果
    int zero_count = 0;
    int positive_count = 0;
    for (int i = 0; i < 100; i++) {
        if (data[i] == 0.0f) zero_count++;
        if (data[i] > 0.0f) positive_count++;
    }
    std::cout << "驗證 (前 100 個): " << zero_count << " 個 0, " 
              << positive_count << " 個正數" << std::endl;
    std::cout << "(預期約 33 個 0, 67 個正數)" << std::endl;

    // 比較 Practice 7 與 Practice 8
    std::cout << "\n--- 效能比較 ---" << std::endl;
    std::cout << "ReLU (Memory Bound): 執行時間短，但 GPU 利用率較低" << std::endl;
    std::cout << "GEMM (Compute Bound): 執行時間長，但 GPU 利用率高" << std::endl;
    std::cout << "使用 nsys profile 觀察兩者的 GPU Utilization 差異" << std::endl;

    cudaFree(data);
    return 0;
}
