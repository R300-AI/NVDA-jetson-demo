/*
 * Practice 2: GPU Warp Divergence 之效能分析
 * 
 * 題目要求：
 * 1. 初始化兩個形狀為 [1, 10^7] 的 A, B 向量
 * 2. 若 tid 為偶數執行 C[tid] = A[tid] + B[tid]
 *    若 tid 為奇數執行 C[tid] = A[tid] - B[tid]
 * 3. 利用 std::chrono 記錄整體執行時間
 * 
 * 編譯: nvcc lab2_divergence.cu -o lab2 -O2 -arch=sm_87
 * 執行: ./lab2
 * 監測: nsys profile -o report_lab2 ./lab2
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Divergent Kernel: 偶數/奇數 thread 執行不同操作
// 這會導致同一個 Warp 內的 thread 走不同分支，造成效能損失
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

// Optimized Kernel: 重新排列任務，避免 Warp Divergence
// 前半段全做加法，後半段全做減法，讓同一 Warp 的 thread 走相同分支
__global__ void optimized_kernel(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;
    
    if (tid < half) {
        // 前半段: 加法 (對應原本的偶數位置)
        C[tid * 2] = A[tid * 2] + B[tid * 2];
    } else if (tid < N) {
        // 後半段: 減法 (對應原本的奇數位置)
        int idx = (tid - half) * 2 + 1;
        if (idx < N) {
            C[idx] = A[idx] - B[idx];
        }
    }
}

int main() {
    const int N = 10000000;  // 10^7
    size_t bytes = N * sizeof(float);
    
    std::cout << "=== Practice 2: Warp Divergence 效能分析 ===" << std::endl;
    std::cout << "向量長度: " << N << " (10^7)" << std::endl;

    // 配置 Managed Memory
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // 初始化
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    divergent_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    // ========== 測試 Divergent Kernel ==========
    for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }
    
    auto start_div = std::chrono::high_resolution_clock::now();
    divergent_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end_div = std::chrono::high_resolution_clock::now();
    double t_div = std::chrono::duration<double>(end_div - start_div).count();

    // ========== 測試 Optimized Kernel ==========
    for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }
    
    // Warmup
    optimized_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }
    
    auto start_opt = std::chrono::high_resolution_clock::now();
    optimized_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end_opt = std::chrono::high_resolution_clock::now();
    double t_opt = std::chrono::duration<double>(end_opt - start_opt).count();

    // ========== 輸出結果 ==========
    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "Divergent Kernel Time: " << t_div << " s" << std::endl;
    std::cout << "Optimized Kernel Time: " << t_opt << " s" << std::endl;
    
    double loss = (t_div - t_opt) / t_div * 100.0;
    std::cout << "效能損失 (Divergence): " << loss << "%" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
