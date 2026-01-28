#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel: 向量加法
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats，觀察:" << std::endl;
    std::cout << "1. GR3D_FREQ (GPU 使用率)" << std::endl;
    std::cout << "2. VDD_GPU 功耗數值" << std::endl;

    // ========== TODO 1: 設定向量大小 ==========
    // 建議使用 10^8 以觀察明顯的效能差異

    int N = 1000;                   /* 請填入正確的向量大小 */
    size_t bytes = N * sizeof(float);
    std::cout << "向量長度: " << N << std::endl;


    // ========== TODO 2: 配置 Managed Memory ==========
    // 使用 cudaMallocManaged 配置 A, B, C 三個向量的記憶體

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);   /* 請接著配置 B, C 的記憶體 */


    // ========== 初始化向量數值 ==========
    for(int i = 0; i < N; i++) { 
        A[i] = 1.0f; 
        B[i] = 2.0f; 
    }


    // ========== CPU 實作 ==========
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " s" << std::endl;


    // ========== TODO 3: 設定 GPU 執行配置 ==========
    // threads: 每個 Block 的執行緒數 (通常為 256)
    // blocks: Block 數量，需確保能涵蓋所有元素

    int threads = 256;
    int blocks = 1;                 /* 請計算正確的 Block 數量 */


    // ========== TODO 4: GPU Warmup ==========
    // GPU 首次執行需要「預熱」，否則計時會不準確
    // 請在計時前先執行一次 kernel

    /* 請加入 Warmup 程式碼 */


    // ========== GPU 計時開始 ==========
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    
    // ========== TODO 5: 執行 GPU Kernel ==========
    // 使用 vector_add_gpu<<<blocks, threads>>>(...) 執行向量加法
    // 記得使用 cudaDeviceSynchronize() 等待 GPU 完成

    /* 請填入 Kernel 執行程式碼 */

    
    // ========== GPU 計時結束 ==========
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " s" << std::endl;


    // ========== 輸出加速倍率 ==========
    std::cout << "加速倍率: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;


    // ========== 釋放記憶體 ==========
    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C);
    
    return 0;
}
