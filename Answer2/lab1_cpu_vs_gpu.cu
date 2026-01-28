#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel: 向量加法
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int N) {
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
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 CUDA Kernel 時間軸" << std::endl;

    const int N = 10000000;  // 10^7
    size_t bytes = N * sizeof(float);
    
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
    vector_add_gpu<<<blocks, threads>>>(A, B, C_gpu, N);
    cudaDeviceSynchronize();

    // ========== CPU 計時 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vector_add_cpu(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " s" << std::endl;

    // ========== GPU 計時 ==========
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vector_add_gpu<<<blocks, threads>>>(A, B, C_gpu, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " s" << std::endl;

    // ========== 輸出結果 ==========
    std::cout << "加速倍率: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cpu);
    cudaFree(C_gpu);
    return 0;
}
