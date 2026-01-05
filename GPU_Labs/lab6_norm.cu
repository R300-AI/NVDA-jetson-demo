#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// 計算 Mean 與 Std 的 Kernel (簡易版: 單一 Block 處理一列，僅作教學示範)
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        
        // 1. 計算總和 (讀取 Global Memory)
        for (int i = 0; i < cols; ++i) {
            float val = data[row * cols + i];
            sum += val;
        }
        float mean = sum / cols;

        // 2. 計算標準差
        for (int i = 0; i < cols; ++i) {
            float diff = data[row * cols + i] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / cols);

        // 3. 正規化寫回 (寫入 Global Memory)
        for (int i = 0; i < cols; ++i) {
            data[row * cols + i] = (data[row * cols + i] - mean) / (std_dev + 1e-5f);
        }
    }
}

int main() {
    int rows = 4096;
    int cols = 768;
    size_t size = rows * cols * sizeof(float);

    float *d_data;
    cudaMallocManaged(&d_data, size);

    // 使用 Eigen 初始化數據
    Eigen::Map<Eigen::MatrixXf> mat(d_data, rows, cols);
    mat.setRandom();

    std::cout << "矩陣大小: " << rows << "x" << cols << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // 每個 Row 分配一個 Block 來處理
    normalize_kernel<<<rows, 1>>>(d_data, rows, cols);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "Normalization 時間: " << diff.count() << " s" << std::endl;

    cudaFree(d_data);
    return 0;
}