#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

// Normalization Kernel: 對每一列進行正規化
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr = data + row * cols;
    
    // Step 1: 計算該列的總和
    float sum = 0.0f;
    /* 請使用 for 迴圈計算該列所有元素的總和
       提示: for (int i = 0; i < cols; i++) { sum += row_ptr[i]; }
    */
    for (int i = 0; i < cols; i++) {
        sum += row_ptr[i];
    }

    float mean = sum / cols;

    // Step 2: 計算標準差
    float sq_sum = 0.0f;
    /* 請計算該列的平方差總和
       提示: for (int i = 0; i < cols; i++) {
                 float diff = row_ptr[i] - mean;
                 sq_sum += diff * diff;
             }
    */
    for (int i = 0; i < cols; i++) {
        float diff = row_ptr[i] - mean;
        sq_sum += diff * diff;
    }

    float std_dev = sqrtf(sq_sum / cols + 1e-5f);

    // Step 3: 正規化
    /* 請將每個元素正規化: (x - mean) / std_dev
       提示: for (int i = 0; i < cols; i++) {
                 row_ptr[i] = (row_ptr[i] - mean) / std_dev;
             }
    */
    for (int i = 0; i < cols; i++) {
        row_ptr[i] = (row_ptr[i] - mean) / std_dev;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile --trace=cuda 監測，觀察 Normalization Kernel" << std::endl;

    // ========== TODO 1: 建立一個隨機浮點數矩陣 A (512 x 768) ==========
    int rows = 512;                 /* 可調整為 2048, 4096, 8192 */
    int cols = 768;
    size_t size = rows * cols * sizeof(float);
    std::cout << "矩陣大小: " << rows << " x " << cols << std::endl;

    float *d_data;
    /* 請使用 cudaMallocManaged(&d_data, size) 配置 Managed Memory */
    cudaMallocManaged(&d_data, size);


    // ========== TODO 2: 初始化矩陣數值 ==========
    /* 請初始化矩陣為隨機值
       提示: std::srand(42);
             fill_random_uniform_0_1(d_data, rows * cols);
    */
    std::srand(42);
    fill_random_uniform_0_1(d_data, rows * cols);

    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // ========== TODO 4: 執行 Normalization Kernel ==========
    /* 請呼叫 normalize_kernel<<<rows, 1>>>(d_data, rows, cols)
       說明: 每個 Block 處理一列，所以 blocks = rows, threads = 1
    */
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
