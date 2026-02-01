#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <chrono>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile --capture-range=cudaProfilerApi 監測" << std::endl;

    // ========== TODO 1: 建立 2048×2048 的大型矩陣 A ==========
    int N = 512;                    /* 請將矩陣大小改為 2048 */
    size_t size = N * N * sizeof(float);
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;

    float *d_A, *d_C;
    cudaMallocManaged(&d_A, size);
    /* 請使用 cudaMallocManaged 配置 d_C 的記憶體 */


    // ========== 初始化矩陣 A ==========
    std::srand(42);
    fill_random_uniform_0_1(d_A, N * N);

    // ========== TODO 2: 調用 cublasSgemm 計算矩陣內積 A·A ==========
    cublasHandle_t handle;
    /* 請呼叫 cublasCreate(&handle) 初始化 cuBLAS */


    // ========== 設定 GEMM 參數 ==========
    float alpha = 1.0f;
    float beta = 0.0f;

    // ========== 開始 Profiling（僅追蹤 GEMM 計算）==========
    cudaProfilerStart();

    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    std::cout << "開始執行 cuBLAS SGEMM (A * A)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    /* 請呼叫 cublasSgemm 計算 C = A * A
       提示: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                         &alpha, d_A, N, d_A, N, &beta, d_C, N)
    */


    cudaDeviceSynchronize();
    
    
    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    

    // ========== 輸出結果 ==========
    std::cout << "執行時間: " << diff.count() << " s" << std::endl;
    

    // ========== TODO 4: 計算 TFLOPS ==========
    // TFLOPS = (2 * N * N * N) / (時間(秒) * 10^12)
    double flops = 2.0 * N * N * N;
    /* 請計算 TFLOPS 並輸出: std::cout << "TFLOPS: " << (flops / (diff.count() * 1e12)) << std::endl; */

    // ========== 停止 Profiling ==========
    cudaProfilerStop();

    // ========== 清理資源 ==========
    cublasDestroy(handle);
    cudaFree(d_A); 
    cudaFree(d_C);
    
    return 0;
}
