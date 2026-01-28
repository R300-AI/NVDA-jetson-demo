/*
 * Practice 1: CPU 與 GPU 的效能差異
 * 
 * 題目要求：
 * 1. 初始化兩個形狀為 [1, 10^7] 的 A, B 向量
 * 2. 計算 A + B
 * 3. 利用 std::chrono 記錄整體執行時間
 * 
 * 編譯: nvcc lab1_cpu_vs_gpu.cu -o lab1 -O2 -arch=sm_87
 * 執行: ./lab1
 * 監測: nsys profile -o report_lab1 ./lab1
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel: 向量加法
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU 向量加法
void vector_add_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10000000;  // 10^7 (約 40MB × 3 = 120MB，安全範圍)
    size_t bytes = N * sizeof(float);
    
    std::cout << "=== Practice 1: CPU vs GPU 向量加法 ===" << std::endl;
    std::cout << "向量長度: " << N << " (10^7)" << std::endl;

    // 配置 Managed Memory
    float *A, *B, *C_cpu, *C_gpu;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C_cpu, bytes);
    cudaMallocManaged(&C_gpu, bytes);

    // 初始化
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // GPU 執行配置
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    vector_add_kernel<<<blocks, threads>>>(A, B, C_gpu, N);
    cudaDeviceSynchronize();

    // ========== CPU 計時 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vector_add_cpu(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // ========== GPU 計時 ==========
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vector_add_kernel<<<blocks, threads>>>(A, B, C_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // ========== 輸出結果 ==========
    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " s" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " s" << std::endl;
    std::cout << "加速倍率 (Speedup): " << cpu_time / gpu_time << "x" << std::endl;

    // 驗證結果
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "驗證: " << (correct ? "✓ 正確" : "✗ 錯誤") << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cpu);
    cudaFree(C_gpu);
    return 0;
}
