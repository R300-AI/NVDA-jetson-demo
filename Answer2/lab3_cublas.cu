/*
 * Practice 3: 利用 cuBLAS 進行高效的線性代數運算
 * 
 * 題目要求：
 * 1. 建立 2048×2048 的大型矩陣 A
 * 2. 調用 cublasSgemm 計算矩陣內積 A·A
 * 3. 利用 std::chrono 記錄整體執行時間
 * 
 * 編譯: nvcc lab3_cublas.cu -o lab3 -O2 -arch=sm_87 -lcublas
 * 執行: ./lab3
 * 監測: nsys profile -o report_lab3 ./lab3
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int N = 2048;
    size_t size = N * N * sizeof(float);
    
    std::cout << "=== Practice 3: cuBLAS 矩陣乘法 ===" << std::endl;
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;

    // 配置 Managed Memory
    float *A, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&C, size);

    // 初始化矩陣 A
    for (int i = 0; i < N * N; i++) {
        A[i] = 0.01f;  // 使用小數值避免溢位
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, A, N, A, N, &beta, C, N);
    cudaDeviceSynchronize();

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();
    
    // C = A × A (矩陣內積)
    // cuBLAS 使用 Column-major，參數順序: (M, N, K)
    // 這裡 M=N=K=2048
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,  // 不轉置
                N, N, N,                    // M, N, K
                &alpha,
                A, N,                       // 矩陣 A, leading dimension
                A, N,                       // 矩陣 A (自己乘自己)
                &beta,
                C, N);                      // 結果矩陣 C
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // ========== 輸出結果 ==========
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "執行時間: " << elapsed << " s" << std::endl;
    
    // 計算 TFLOPS = (2 × N × N × N) / (執行時間 × 10^12)
    double flops = 2.0 * N * N * N;
    double tflops = flops / (elapsed * 1e12);
    std::cout << "TFLOPS: " << tflops << std::endl;
    
    // 驗證結果 (每個元素應為 N × 0.01 × 0.01 = 0.0001 × N)
    float expected = N * 0.01f * 0.01f;
    std::cout << "驗證: C[0] = " << C[0] << " (預期約 " << expected << ")" << std::endl;

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(C);
    return 0;
}
