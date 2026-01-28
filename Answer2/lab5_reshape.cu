/*
 * Practice 5: 透過指標減少 Reshape 的資料搬移
 * 
 * 題目要求：
 * 1. 利用 Eigen::MatrixXf::Random(1024, 768) 在 CPU 建立一個隨機浮點數矩陣 A
 * 2. 使用 reshaped() 將其轉換為 (768×1024) 向量
 * 3. 利用指標映射至 GPU
 * 4. 計算該向量的內積 A·A
 * 
 * 編譯: nvcc lab5_reshape.cu -o lab5 -O2 -arch=sm_87 -I/usr/include/eigen3
 * 執行: ./lab5
 * 監測: nsys profile -o report_lab5 ./lab5
 */

#include <iostream>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// 內積 Kernel (使用 atomicAdd)
__global__ void dot_product_kernel(const float* vec, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, vec[idx] * vec[idx]);
    }
}

int main() {
    const int rows = 1024;
    const int cols = 768;
    const int N = rows * cols;
    size_t size = N * sizeof(float);

    std::cout << "=== Practice 5: Zero-Copy Reshape ===" << std::endl;
    std::cout << "原始矩陣: " << rows << " x " << cols << std::endl;
    std::cout << "Reshape 後: " << N << " x 1 向量" << std::endl;

    // 使用 Managed Memory 配置記憶體
    float* ptr;
    cudaMallocManaged(&ptr, size);

    // 使用 Eigen::Map 建立矩陣視圖 (Zero-Copy: 不會複製資料)
    Eigen::Map<Eigen::MatrixXf> mat(ptr, rows, cols);
    mat.setRandom();  // 填入隨機數

    // 印出原始矩陣首位址
    std::cout << "\n--- 位址驗證 ---" << std::endl;
    std::cout << "原始矩陣首位址: " << static_cast<void*>(ptr) << std::endl;

    // Reshape 為一維向量 (使用 Eigen::Map，不會複製資料)
    Eigen::Map<Eigen::VectorXf> vec(ptr, N);
    
    // 印出 Reshape 後的首位址
    std::cout << "Reshape 後首位址: " << static_cast<void*>(&vec(0)) << std::endl;

    // 驗證位址相同
    if (ptr == &vec(0)) {
        std::cout << "✓ 驗證成功：位址相同，無資料搬移 (Zero-Copy)" << std::endl;
    } else {
        std::cout << "✗ 驗證失敗：位址不同，發生資料拷貝" << std::endl;
    }

    // 在 GPU 上計算內積 (同一塊記憶體，無需 cudaMemcpy)
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0.0f;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    std::cout << "\n--- GPU 內積計算 ---" << std::endl;
    dot_product_kernel<<<blocks, threads>>>(ptr, d_result, N);
    cudaDeviceSynchronize();

    std::cout << "內積結果: " << *d_result << std::endl;

    // 使用 Eigen 在 CPU 驗證
    float cpu_result = vec.squaredNorm();
    std::cout << "CPU 驗證: " << cpu_result << std::endl;
    
    float diff = fabsf(*d_result - cpu_result) / cpu_result;
    std::cout << "誤差: " << diff * 100 << "%" << std::endl;

    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}
