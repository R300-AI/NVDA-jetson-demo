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
        
        /* 請完成 bias 加法: C[idx] = C[idx] + b[row] */
        
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 GEMM vs Bias 時間佔比" << std::endl;

    // ========== TODO 1: 使用 cuBLAS 執行 2048×2048 的矩陣乘法 ==========
    int M = 1024;                   /* 請填入正確的矩陣大小 (2048) */
    int N = 1024;
    int K = 1024;
    std::cout << "矩陣大小: " << M << " x " << N << std::endl;
    
    size_t mat_size = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_b;
    cudaMallocManaged(&d_A, mat_size);
    /* 請接著配置 d_B, d_C, d_b 的記憶體 */


    // ========== 初始化數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(d_A, M * N);
    fill_random_uniform_0_1(d_B, M * N);
    fill_random_uniform_0_1(d_b, M);

    // ========== TODO 2: 建立 cuBLAS Handle ==========
    cublasHandle_t handle;
    /* 請初始化 cuBLAS handle */


    float alpha = 1.0f, beta = 0.0f;

    // ========== 開始計時 ==========
    std::cout << "開始執行 GEMM + Bias..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // ========== TODO 3: 執行 GEMM (C = A × B) ==========

    /* 請填入 GEMM 執行程式碼 */

    cudaDeviceSynchronize();

    // ========== TODO 4: 執行 Bias Addition (C' = C + b) ==========
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    
    /* 請執行 bias_add_kernel */

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
