#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// ========== TODO: 實作 Bias Addition Kernel ==========
// 將偏置向量 b 加到矩陣 C 的每一列
// C'[i][j] = C[i][j] + b[i]
__global__ void bias_add_kernel(float* C, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int row = idx / N;          // 計算該元素屬於哪一列
        
        /* 請完成 bias 加法 */
        
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats，觀察:" << std::endl;
    std::cout << "1. VDD_GPU 功耗數值 (GEMM 時應飆升)" << std::endl;
    std::cout << "2. GR3D_FREQ (GPU 使用率)" << std::endl;

    // ========== TODO 1: 設定矩陣大小 ==========
    // 題目要求 2048 x 2048

    int M = 1024;                   /* 請填入正確的矩陣大小 */
    int N = 1024;
    int K = 1024;
    
    std::cout << "矩陣大小: " << M << " x " << N << std::endl;
    

    // ========== TODO 2: 配置記憶體 ==========
    // d_A, d_B: 輸入矩陣
    // d_C: 結果矩陣 (GEMM 輸出)
    // d_b: 偏置向量 (長度為 M)

    float *d_A, *d_B, *d_C, *d_b;
    size_t mat_size = M * N * sizeof(float);
    
    cudaMallocManaged(&d_A, mat_size);
    /* 請接著配置 d_B, d_C, d_b 的記憶體 */


    // ========== 初始化數值 ==========
    for(int i = 0; i < M * N; i++) { 
        d_A[i] = 1.0f; 
        d_B[i] = 0.5f; 
    }
    for(int i = 0; i < M; i++) {
        d_b[i] = 0.1f;              // 偏置值
    }


    // ========== TODO 3: 建立 cuBLAS Handle ==========

    cublasHandle_t handle;
    /* 請初始化 cuBLAS handle */


    float alpha = 1.0f, beta = 0.0f;


    // ========== 開始計時 ==========
    std::cout << "開始執行 GEMM + Bias..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();


    // ========== TODO 4: 執行 GEMM (C = A × B) ==========
    // 使用 cublasSgemm

    /* 請填入 GEMM 執行程式碼 */


    // ========== TODO 5: 執行 Bias Addition (C' = C + b) ==========
    // 使用 bias_add_kernel

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    
    /* 請執行 bias_add_kernel */


    cudaDeviceSynchronize();


    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    // ========== 輸出結果 ==========
    std::cout << "GEMM + Bias 執行時間: " << diff.count() << " s" << std::endl;


    // ========== 清理資源 ==========
    cublasDestroy(handle);
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    cudaFree(d_b);
    
    return 0;
}
