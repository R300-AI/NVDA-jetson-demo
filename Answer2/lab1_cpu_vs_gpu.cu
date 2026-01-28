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
    int N = 100000000; // 10^8
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;

    // 使用 Managed Memory (Jetson 上 CPU/GPU 共用記憶體)
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // 初始化
    for(int i = 0; i < N; i++) { 
        A[i] = 1.0f; 
        B[i] = 2.0f; 
    }

    // ========== CPU 實作 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " s" << std::endl;

    // ========== GPU 實作 ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Warmup - 讓 GPU 從休眠中喚醒
    vector_add_gpu<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    vector_add_gpu<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " s" << std::endl;

    // 計算加速倍率
    std::cout << "加速倍率: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;

    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    return 0;
}
