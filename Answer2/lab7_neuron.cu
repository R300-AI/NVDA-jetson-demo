#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

// Bias Addition Kernel
__global__ void bias_add_kernel(float* C, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int row = idx / N;
        C[idx] = C[idx] + b[row];
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 GEMM vs Bias 時間佔比" << std::endl;

    // ========== TODO 1: 使用 cuBLAS 執行 2048×2048 的矩陣乘法 ==========
    int M = 2048;
    int N = 2048;
    int K = 2048;
    std::cout << "矩陣大小: " << M << " x " << N << std::endl;
    
    size_t mat_size = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_b;
    cudaMallocManaged(&d_A, mat_size);
    cudaMallocManaged(&d_B, mat_size);
    cudaMallocManaged(&d_C, mat_size);
    cudaMallocManaged(&d_b, M * sizeof(float));

    // ========== 初始化數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(d_A, M * N);
    fill_random_uniform_0_1(d_B, M * N);
    fill_random_uniform_0_1(d_b, M);

    // ========== TODO 2: 建立 cuBLAS Handle ==========
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // ========== 開始計時 ==========
    std::cout << "開始執行 GEMM + Bias..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // ========== TODO 3: 執行 GEMM (C = A × B) ==========
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();

    // ========== TODO 4: 執行 Bias Addition (C' = C + b) ==========
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(d_C, d_b, M, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ========== 輸出結果 ==========
    std::cout << "GEMM + Bias 執行時間: " << diff.count() << " s" << std::endl;

    // ========== 釋放記憶體 ==========
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_b);
    return 0;
}
