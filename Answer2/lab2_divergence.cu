#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

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

// Optimized Kernel: 重新排列任務，避免 Warp Divergence
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
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 Warp Stall Reasons" << std::endl;

    const int N = 10000000;  // 10^7
    size_t bytes = N * sizeof(float);
    
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
    std::chrono::duration<double> t_div = end_div - start_div;

    // ========== 測試 Optimized Kernel ==========
    for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }
    
    optimized_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }
    
    auto start_opt = std::chrono::high_resolution_clock::now();
    optimized_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_opt = end_opt - start_opt;

    // ========== 輸出結果 ==========
    std::cout << "Divergent Time: " << t_div.count() << " s" << std::endl;
    std::cout << "Optimized Time: " << t_opt.count() << " s" << std::endl;
    
    double loss = (t_div.count() - t_opt.count()) / t_div.count() * 100.0;
    std::cout << "效能損失 (Divergence): " << loss << "%" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
