#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// Bias Addition Kernel
// 將偏置向量 b 加到矩陣 C 的每一列
// C'[i][j] = C[i][j] + b[i]
__global__ void bias_add_kernel(float* C, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int row = idx / N;  // 計算該元素屬於哪一列
        C[idx] = C[idx] + b[row];
    }
}

int main() {
    int M = 2048;
    int N = 2048;
    int K = 2048;
    
    std::cout << "矩陣大小: " << M << " x " << N << std::endl;

    // 配置記憶體
    float *d_A, *d_B, *d_C, *d_b;
    size_t mat_size = M * N * sizeof(float);
    
    cudaMallocManaged(&d_A, mat_size);
    cudaMallocManaged(&d_B, mat_size);
    cudaMallocManaged(&d_C, mat_size);
    cudaMallocManaged(&d_b, M * sizeof(float));

    // 初始化數值
    for(int i = 0; i < M * N; i++) { 
        d_A[i] = 1.0f; 
        d_B[i] = 0.5f; 
    }
    for(int i = 0; i < M; i++) {
        d_b[i] = 0.1f;  // 偏置值
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // 計時開始
    std::cout << "開始執行 GEMM + Bias..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: GEMM (C = A × B)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, d_B, N, d_A, K, 
                &beta, d_C, N);

    // Step 2: Bias Addition (C' = C + b)
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(d_C, d_b, M, N);

    cudaDeviceSynchronize();

    // 計時結束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "GEMM + Bias 執行時間: " << diff.count() << " s" << std::endl;

    // 驗證結果
    std::cout << "驗證：C'[0] = " << d_C[0] << " (預期: " << (K * 0.5f + 0.1f) << ")" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    cudaFree(d_b);
    return 0;
}
