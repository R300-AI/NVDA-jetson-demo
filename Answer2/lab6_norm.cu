#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cmath>

// Normalization Kernel
// 對每一列進行正規化: (x - mean) / std
// 每個 Block 處理一列
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        // Step 1: 計算該列的總和
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sum += data[row * cols + i];
        }
        float mean = sum / cols;

        // Step 2: 計算標準差
        float sq_sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float diff = data[row * cols + i] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / cols);

        // Step 3: 正規化並寫回
        for (int i = 0; i < cols; ++i) {
            data[row * cols + i] = (data[row * cols + i] - mean) / (std_dev + 1e-5f);
        }
    }
}

int main() {
    int rows = 512;  // 可調整為 2048, 4096, 8192
    int cols = 768;
    size_t size = rows * cols * sizeof(float);

    std::cout << "矩陣大小: " << rows << " x " << cols << std::endl;

    // 配置 Managed Memory
    float *d_data;
    cudaMallocManaged(&d_data, size);

    // 使用 Eigen 初始化隨機數據
    Eigen::Map<Eigen::MatrixXf> mat(d_data, rows, cols);
    mat.setRandom();

    // Warmup
    normalize_kernel<<<rows, 1>>>(d_data, rows, cols);
    cudaDeviceSynchronize();

    // 重新初始化
    mat.setRandom();

    // 計時開始
    auto start = std::chrono::high_resolution_clock::now();

    // 執行 Normalization Kernel
    // 每一列分配一個 Block
    normalize_kernel<<<rows, 1>>>(d_data, rows, cols);
    cudaDeviceSynchronize();

    // 計時結束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Normalization 時間: " << diff.count() << " s" << std::endl;

    // 驗證結果 (檢查第一列的 mean 應接近 0)
    float check_sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        check_sum += d_data[i];
    }
    std::cout << "驗證：第一列 Mean (應接近 0): " << check_sum / cols << std::endl;

    cudaFree(d_data);
    return 0;
}
