#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

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

    // ========== TODO 1: 設定向量大小與配置記憶體 ==========
    int N = 10000000;               /* 請將向量大小改為 10000000 (10^7) */
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;

    float *A, *B, *C_cpu, *C_gpu;
    cudaMallocManaged(&A, bytes);
    /* 請使用 cudaMallocManaged 配置 B, C_cpu, C_gpu 的記憶體 */
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C_cpu, bytes);
    cudaMallocManaged(&C_gpu, bytes);

    // ========== 初始化向量數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A, N);
    fill_random_uniform_0_1(B, N);

    // ========== CPU 加法測試 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vector_add_cpu(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // ========== TODO 2: GPU 加法測試 ==========
    // 設定執行配置: threads (每個 Block 的執行緒數), blocks (Block 數量)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;   /* 請計算正確的 Block 數量: (N + threads - 1) / threads */

    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    /* 請呼叫 vector_add_gpu<<<blocks, threads>>>(A, B, C_gpu, N) 執行向量加法 */
    /* 請呼叫 cudaDeviceSynchronize() 等待 GPU 完成 */
    vector_add_gpu<<<blocks, threads>>>(A, B, C_gpu, N);
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // ========== 輸出結果 ==========
    std::cout << "CPU Time: " << cpu_time.count() << " s" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " s" << std::endl;
    std::cout << "加速倍率: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;

    // ========== 釋放記憶體 ==========
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cpu);
    cudaFree(C_gpu);
    return 0;
}
