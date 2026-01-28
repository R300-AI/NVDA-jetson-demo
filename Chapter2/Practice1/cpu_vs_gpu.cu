#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel: 向量加法
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 CUDA Kernel 時間軸" << std::endl;

    // ========== TODO 1: 初始化兩個形狀為 [1, 10^7] 的 A, B 向量 ==========
    // 使用 cudaMallocManaged 配置 A, B, C 三個向量的記憶體

    int N = 1000;                   /* 請填入正確的向量大小 (10^7) */
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);   /* 請接著配置 B, C 的記憶體 */


    // ========== 初始化向量數值 ==========
    for(int i = 0; i < N; i++) { 
        A[i] = 1.0f; 
        B[i] = 2.0f; 
    }


    // ========== TODO 2: 分別使用 CPU for-loop 與 GPU Kernel 計算 A + B ==========

    // --- CPU 實作 ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " s" << std::endl;

    // --- GPU 實作 ---
    // 設定執行配置: threads (每個 Block 的執行緒數), blocks (Block 數量)
    int threads = 256;
    int blocks = 1;                 /* 請計算正確的 Block 數量: (N + threads - 1) / threads */

    // GPU Warmup (首次執行需要「預熱」)
    /* 請加入 Warmup 程式碼 */

    // GPU 計時
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    /* 請使用 vector_add_gpu<<<blocks, threads>>>(...) 執行向量加法 */
    /* 記得使用 cudaDeviceSynchronize() 等待 GPU 完成 */

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " s" << std::endl;


    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    // 計算並輸出加速倍率 (Speedup Ratio) = CPU_time / GPU_time

    std::cout << "加速倍率: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;


    // ========== 釋放記憶體 ==========
    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    
    return 0;
}
