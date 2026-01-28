#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// Normalization Kernel: 對每一列進行正規化
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr = data + row * cols;
    
    // Step 1: 計算該列的總和
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum += row_ptr[i];
    }
    float mean = sum / cols;

    // Step 2: 計算標準差
    float sq_sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float diff = row_ptr[i] - mean;
        sq_sum += diff * diff;
    }
    float std_dev = sqrtf(sq_sum / cols + 1e-5f);

    // Step 3: 正規化
    for (int i = 0; i < cols; i++) {
        row_ptr[i] = (row_ptr[i] - mean) / std_dev;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 Memory Throughput" << std::endl;

    // ========== TODO 1: 建立一個隨機浮點數矩陣 A (512 x 768) ==========
    int rows = 512;
    int cols = 768;
    size_t size = rows * cols * sizeof(float);
    std::cout << "矩陣大小: " << rows << " x " << cols << std::endl;

    float *d_data;
    cudaMallocManaged(&d_data, size);

    // ========== TODO 2: 使用 Eigen::Map 初始化矩陣 ==========
    Eigen::Map<Eigen::MatrixXf> mat(d_data, rows, cols);
    mat.setRandom();

    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // ========== TODO 4: 執行 Normalization Kernel ==========
    normalize_kernel<<<rows, 1>>>(d_data, rows, cols);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ========== 輸出結果 ==========
    std::cout << "Normalization 時間: " << diff.count() << " s" << std::endl;

    // ========== 釋放記憶體 ==========
    cudaFree(d_data);
    return 0;
}
