#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// 簡單的內積 Kernel
__global__ void dot_product_kernel(const float* vec, float* result, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* 請實作內積 Kernel */
    /* 提示: 使用 atomicAdd(result, vec[idx] * vec[idx]) */
    
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察是否有 cudaMemcpy" << std::endl;

    // ========== TODO 1: 建立一個隨機浮點數矩陣 A (1024 x 768) ==========
    int rows = 512;                 /* 請填入正確的行數 (1024) */
    int cols = 256;                 /* 請填入正確的列數 (768) */
    size_t size = rows * cols * sizeof(float);

    float* ptr;
    /* 請配置 Managed Memory */


    // ========== TODO 2: 使用 Eigen::Map 建立矩陣視圖 ==========

    /* 請建立 Eigen::Map 矩陣並填入隨機值 */


    // ========== 印出原始矩陣首位址 ==========
    std::cout << "原始矩陣首位址: " << ptr << std::endl;

    // ========== TODO 3: 將其視為一維向量（不複製資料）==========

    /* 請建立 Reshape 後的向量視圖 */


    // ========== 驗證位址相同 ==========

    /* 請印出 Reshape 後的首位址並驗證 */


    // ========== TODO 4: 實作 GPU Kernel 計算該向量的內積 A·A ==========
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0;

    int threads = 512;
    int blocks = (rows * cols + threads - 1) / threads;

    /* 請執行 Kernel 並輸出結果 */


    // ========== 輸出結果 ==========

    /* 請輸出 GPU 內積結果 */


    // ========== 釋放記憶體 ==========
    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}
