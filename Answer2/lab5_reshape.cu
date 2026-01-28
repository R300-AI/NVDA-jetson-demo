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

    const int rows = 1024;
    const int cols = 768;
    size_t size = rows * cols * sizeof(float);

    float* ptr;
    cudaMallocManaged(&ptr, size);

    // 使用 Eigen::Map 建立矩陣視圖
    Eigen::Map<Eigen::MatrixXf> mat(ptr, rows, cols);
    mat.setRandom();

    // 印出原始矩陣首位址
    std::cout << "原始矩陣首位址: " << ptr << std::endl;

    // 使用 Eigen::Map 將相同指標映射為向量
    Eigen::Map<Eigen::VectorXf> vec(ptr, rows * cols);

    // 驗證位址相同
    std::cout << "Reshape 後首位址: " << vec.data() << std::endl;

    // GPU 計算內積
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0;

    int threads = 512;
    int blocks = (rows * cols + threads - 1) / threads;

    dot_product_kernel<<<blocks, threads>>>(ptr, d_result, rows * cols);
    cudaDeviceSynchronize();

    std::cout << "GPU 內積結果: " << *d_result << std::endl;

    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}
