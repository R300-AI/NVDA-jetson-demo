#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 cublasSgemm 執行時間" << std::endl;

    const int N = 2048;
    size_t size = N * N * sizeof(float);
    
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;

    // 配置 Managed Memory
    float *d_A, *d_C;
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_C, size);

    // 初始化矩陣 A
    for (int i = 0; i < N * N; i++) {
        d_A[i] = 1.0f;
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    // ========== 計時開始 ==========
    std::cout << "開始執行 cuBLAS SGEMM (A * A)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_A, N,
                &beta,
                d_C, N);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ========== 輸出結果 ==========
    std::cout << "執行時間: " << diff.count() << " s" << std::endl;
    
    // 計算 TFLOPS
    double flops = 2.0 * N * N * N;
    double tflops = flops / (diff.count() * 1e12);
    std::cout << "TFLOPS: " << tflops << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_C);
    return 0;
}
