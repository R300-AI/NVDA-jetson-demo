/*
 * Practice 6: Normalization 的記憶體層級差異
 * 
 * 題目要求：
 * 1. 利用 Eigen::MatrixXf::Random(512, 768) 在 CPU 建立一個隨機浮點數矩陣 A
 * 2. 計算 Mean 與 Std，並計算 (A−Mean)/Std
 * 3. 利用 std::chrono 記錄整體執行時間
 * 
 * 編譯: nvcc lab6_norm.cu -o lab6 -O2 -arch=sm_87 -I/usr/include/eigen3
 * 執行: ./lab6
 * 監測: nsys profile -o report_lab6 ./lab6
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// Normalization Kernel: 對每一列進行正規化
// 每個 Block 處理一列: (x - mean) / std
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr = data + row * cols;
    
    // Step 1: 計算該列的平均值
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

void run_test(int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    // 配置 Managed Memory
    float* data;
    cudaMallocManaged(&data, size);

    // 使用 Eigen 初始化隨機數據
    Eigen::Map<Eigen::MatrixXf> mat(data, rows, cols);
    mat.setRandom();

    // Warmup
    normalize_kernel<<<rows, 1>>>(data, rows, cols);
    cudaDeviceSynchronize();

    // 重新初始化
    mat.setRandom();

    // 計時
    auto start = std::chrono::high_resolution_clock::now();
    normalize_kernel<<<rows, 1>>>(data, rows, cols);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    // 驗證 (正規化後每列 mean 應接近 0)
    float check_sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        check_sum += data[i];
    }
    float row0_mean = check_sum / cols;

    std::cout << "矩陣 " << rows << " x " << cols 
              << " | 時間: " << elapsed * 1000 << " ms"
              << " | 第一列 Mean: " << row0_mean << std::endl;

    cudaFree(data);
}

int main() {
    const int cols = 768;
    
    std::cout << "=== Practice 6: Normalization 效能分析 ===" << std::endl;
    std::cout << "固定 cols = " << cols << "，逐步放大 rows\n" << std::endl;

    // 題目要求: 512 → 2048 → 4096 → 8192
    run_test(512, cols);
    run_test(2048, cols);
    run_test(4096, cols);
    run_test(8192, cols);

    std::cout << "\n觀察重點:" << std::endl;
    std::cout << "1. 時間是否呈線性增長?" << std::endl;
    std::cout << "2. 使用 nsys profile 觀察 Memory Throughput" << std::endl;
    std::cout << "3. 觀察 Kernel Launch Overhead" << std::endl;

    return 0;
}
