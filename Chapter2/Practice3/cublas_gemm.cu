#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats，觀察:" << std::endl;
    std::cout << "1. GR3D_FREQ (GPU 使用率)" << std::endl;
    std::cout << "2. VDD_GPU 功耗數值" << std::endl;

    // ========== TODO 1: 設定矩陣大小 ==========
    // 題目要求 2048 x 2048

    int N = 512;                    /* 請填入正確的矩陣大小 */
    size_t size = N * N * sizeof(float);
    std::cout << "矩陣大小: " << N << "x" << N << std::endl;


    // ========== TODO 2: 配置 Managed Memory ==========
    // 需要配置矩陣 A 和結果矩陣 C

    float *d_A, *d_C;
    cudaMallocManaged(&d_A, size);  /* 請接著配置 d_C 的記憶體 */


    // ========== 初始化矩陣 A ==========
    for (int i = 0; i < N * N; ++i) {
        d_A[i] = 1.0f;
    }


    // ========== TODO 3: 建立 cuBLAS Handle ==========
    // 使用 cublasCreate() 初始化 cuBLAS

    cublasHandle_t handle;
    /* 請初始化 cuBLAS handle */


    // ========== 設定 GEMM 參數 ==========
    float alpha = 1.0f;
    float beta = 0.0f;


    // ========== TODO 4: Warmup ==========
    // 首次執行 cuBLAS 會有初始化延遲，需要先預熱

    /* 請加入 Warmup 程式碼 */


    // ========== 開始計時 ==========
    std::cout << "開始執行 cuBLAS SGEMM (A * A)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();


    // ========== TODO 5: 執行矩陣乘法 ==========
    // 使用 cublasSgemm 計算 C = A * A
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    //             N, N, N, &alpha, d_A, N, d_A, N, &beta, d_C, N);

    /* 請填入 cublasSgemm 執行程式碼 */


    cudaDeviceSynchronize();
    
    
    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    

    // ========== 輸出結果 ==========
    std::cout << "執行時間: " << diff.count() << " s" << std::endl;
    

    // ========== TODO 6: 計算 TFLOPS ==========
    // TFLOPS = (2 * N * N * N) / (時間 * 10^12)

    /* 請計算並輸出 TFLOPS */


    // ========== 清理資源 ==========
    cublasDestroy(handle);
    cudaFree(d_A); 
    cudaFree(d_C);
    
    return 0;
}
