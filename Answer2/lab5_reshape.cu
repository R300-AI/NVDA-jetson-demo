#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// 簡單的內積 Kernel
__global__ void dot_product_kernel(const float* vec, float* result, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        atomicAdd(result, vec[idx] * vec[idx]);
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察是否有 cudaMemcpy" << std::endl;

    // ========== TODO 1: 建立一個隨機浮點數矩陣 A (1024 x 768) ==========
    int rows = 1024;
    int cols = 768;
    size_t size = rows * cols * sizeof(float);

    float* ptr;
    cudaMallocManaged(&ptr, size);

    // ========== TODO 2: 使用 Eigen::Map 建立矩陣視圖 ==========
    Eigen::Map<Eigen::MatrixXf> mat(ptr, rows, cols);
    mat.setRandom();

    // ========== 印出原始矩陣首位址 ==========
    std::cout << "原始矩陣首位址: " << ptr << std::endl;

    // ========== TODO 3: 將其視為一維向量（不複製資料）==========
    Eigen::Map<Eigen::VectorXf> vec(ptr, rows * cols);

    // ========== 驗證位址相同 ==========
    std::cout << "Reshape 後首位址: " << vec.data() << std::endl;

    // ========== TODO 4: 實作 GPU Kernel 計算該向量的內積 A·A ==========
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0;

    int threads = 512;
    int blocks = (rows * cols + threads - 1) / threads;

    dot_product_kernel<<<blocks, threads>>>(ptr, d_result, rows * cols);
    cudaDeviceSynchronize();

    // ========== 輸出結果 ==========
    std::cout << "GPU 內積結果: " << *d_result << std::endl;

    // ========== 釋放記憶體 ==========
    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}
