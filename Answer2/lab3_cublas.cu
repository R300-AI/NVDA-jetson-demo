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

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 cublasSgemm 執行時間" << std::endl;

    // ========== TODO 1: 建立 2048×2048 的大型矩陣 A ==========
    int N = 2048;
    size_t size = N * N * sizeof(float);
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;

    float *d_A, *d_C;
    cudaMallocManaged(&d_A, size);
    cudaMallocManaged(&d_C, size);

    // ========== 初始化矩陣 A ==========
    std::srand(42);
    fill_random_uniform_0_1(d_A, N * N);

    // ========== TODO 2: 調用 cublasSgemm 計算矩陣內積 A·A ==========
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
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
