#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

int main() {
    // 設定矩陣大小
    int M = 2048, N = 2048, K = 2048;
    std::cout << "[Lab 7] 初始化記憶體 (2048x2048)..." << std::endl;

    float *d_A, *d_B, *d_C;
    size_t mat_size = M * N * sizeof(float);
    
    // 使用 Managed Memory
    cudaMallocManaged(&d_A, mat_size);
    cudaMallocManaged(&d_B, mat_size);
    cudaMallocManaged(&d_C, mat_size);

    // 初始化數值
    for(int i=0; i<M*N; i++) { d_A[i] = 1.0f; d_B[i] = 0.5f; }

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    std::cout << "準備執行 cuBLAS GEMM (矩陣乘法)..." << std::endl;
    std::cout << ">> 請準備好觀察 Tegrastats 的 VDD_IN (功耗應飆升)" << std::endl;
    std::cout << ">> 按下 Enter 開始...";
    std::cin.get();

    std::cout << "正在執行 200 次 GEMM..." << std::endl;
    
    // 執行多次以維持高負載狀態
    for(int i=0; i<200; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    
    cudaDeviceSynchronize();
    std::cout << "GEMM 運算完成。" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}