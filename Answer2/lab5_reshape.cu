#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>

// 簡單的內積 Kernel
__global__ void dot_product_kernel(const float* vec, float* result, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len) {
        atomicAdd(result, vec[idx] * vec[idx]);
    }
}

int main() {
    int rows = 1024;
    int cols = 768;
    size_t size = rows * cols * sizeof(float);

    std::cout << "矩陣大小: " << rows << " x " << cols << std::endl;

    // 使用 Managed Memory 配置記憶體
    float* ptr;
    cudaMallocManaged(&ptr, size);

    // 使用 Eigen::Map 建立矩陣視圖 (不會拷貝資料)
    Eigen::Map<Eigen::MatrixXf> mat(ptr, rows, cols);
    mat.setRandom();

    // 印出原始矩陣首位址
    std::cout << "原始矩陣首位址: " << ptr << std::endl;

    // Reshape 為一維向量 (使用 Eigen::Map，不會拷貝資料)
    Eigen::Map<Eigen::VectorXf> vec(ptr, rows * cols);
    
    // 印出 Reshape 後的首位址
    std::cout << "Reshape 後首位址: " << &vec(0) << std::endl;

    // 驗證位址相同
    if (ptr == &vec(0)) {
        std::cout << "✓ 驗證成功：位址相同，無資料搬移 (Zero-Copy)" << std::endl;
    } else {
        std::cout << "✗ 驗證失敗：位址不同，發生資料拷貝" << std::endl;
    }

    // 在 GPU 上計算內積
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0;

    int threads = 512;
    int blocks = (rows * cols + threads - 1) / threads;

    dot_product_kernel<<<blocks, threads>>>(ptr, d_result, rows * cols);
    cudaDeviceSynchronize();

    std::cout << "內積結果 (部分): " << *d_result << std::endl;

    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}
