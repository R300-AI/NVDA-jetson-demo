#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

int main() {
    int N = 2048; // 矩陣大小
    size_t size = N * N * sizeof(float);
    std::cout << "矩陣大小: " << N << "x" << N << std::endl;

    float *d_A, *d_C;
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_C, size);

    // 初始化 A 矩陣
    for (int i = 0; i < N * N; ++i) d_A[i] = 1.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "開始執行 cuBLAS SGEMM (A * A)..." << std::endl;
    
    // 預熱
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    
    // 計算 C = A * A
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, d_A, N, d_A, N, 
                &beta, d_C, N);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "執行時間: " << diff.count() << " s" << std::endl;
    std::cout << "TFLOPS: " << (2.0 * N * N * N) / (diff.count() * 1e12) << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_C);
    return 0;
}