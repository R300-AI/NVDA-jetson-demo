/*
 * Practice 7: 透過異質運算實踐 Artificial Neurons
 * 
 * 題目要求：
 * 1. 使用 cuBLAS 執行一個 2048×2048 的矩陣乘法 (GEMM)，得到結果矩陣 C
 * 2. 建立一個長度為 2048 的偏置向量 b
 * 3. 實作 Bias Addition Kernel，將偏置向量加到矩陣 C 的每一列：C' = C + b
 * 
 * 編譯: nvcc lab7_neuron.cu -o lab7 -O2 -arch=sm_87 -lcublas
 * 執行: ./lab7
 * 監測: nsys profile -o report_lab7 ./lab7
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Bias Addition Kernel
// 將偏置向量 b 加到矩陣 C 的每一列
// C'[i][j] = C[i][j] + b[i]
__global__ void bias_add_kernel(float* C, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;  // 計算該元素屬於哪一列
        C[idx] = C[idx] + b[row];
    }
}

int main() {
    const int N = 2048;  // 矩陣大小 2048 x 2048
    size_t mat_size = N * N * sizeof(float);
    size_t vec_size = N * sizeof(float);
    
    std::cout << "=== Practice 7: GEMM + Bias (Artificial Neuron) ===" << std::endl;
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;

    // 配置記憶體
    float *A, *B, *C, *b;
    cudaMallocManaged(&A, mat_size);
    cudaMallocManaged(&B, mat_size);
    cudaMallocManaged(&C, mat_size);
    cudaMallocManaged(&b, vec_size);

    // 初始化
    for (int i = 0; i < N * N; i++) {
        A[i] = 0.01f;  // 輸入
        B[i] = 0.01f;  // 權重
    }
    for (int i = 0; i < N; i++) {
        b[i] = 0.1f;   // 偏置
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, B, N, A, N, &beta, C, N);
    cudaDeviceSynchronize();

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: GEMM (C = A × B)
    // 模擬神經網路的線性變換: output = input × weight
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                B, N,       // Weight matrix
                A, N,       // Input matrix
                &beta,
                C, N);      // Output matrix
    cudaDeviceSynchronize();

    // Step 2: Bias Addition (C' = C + b)
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(C, b, N, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // ========== 輸出結果 ==========
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "GEMM + Bias 總時間: " << elapsed << " s" << std::endl;

    // 驗證結果
    // C[0] = sum(A[0,:] * B[:,0]) + b[0]
    //      = N * 0.01 * 0.01 + 0.1
    //      = N * 0.0001 + 0.1
    float expected = N * 0.0001f + 0.1f;
    std::cout << "驗證: C'[0] = " << C[0] << " (預期: " << expected << ")" << std::endl;

    std::cout << "\n記錄此執行時間，供 Practice 8 比較!" << std::endl;

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(b);
    return 0;
}
