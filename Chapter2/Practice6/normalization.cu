#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cmath>

// ========== TODO: 實作 Normalization Kernel ==========
// 對每一列進行正規化: (x - mean) / std
// 每個 Block 處理一列
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        // ========== Step 1: 計算該列的總和 ==========
        float sum = 0.0f;
        
        /* 請計算該列所有元素的總和 */

        float mean = sum / cols;


        // ========== Step 2: 計算標準差 ==========
        float sq_sum = 0.0f;
        
        /* 請計算該列的平方差總和 */

        float std_dev = sqrtf(sq_sum / cols);


        // ========== Step 3: 正規化並寫回 ==========
        
        /* 請將每個元素正規化: (x - mean) / (std + epsilon) */
        /* epsilon = 1e-5f 用於避免除以零 */

    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 Memory Throughput" << std::endl;

    // ========== TODO 1: 建立一個隨機浮點數矩陣 A (512 x 768) ==========
    // 可調整為 2048, 4096, 8192 觀察效能變化

    int rows = 512;                 /* 可調整為 2048, 4096, 8192 */
    int cols = 768;
    size_t size = rows * cols * sizeof(float);
    
    std::cout << "矩陣大小: " << rows << " x " << cols << std::endl;

    float *d_data;
    /* 請配置 Managed Memory */


    // ========== TODO 2: 實作 GPU Kernel 計算 Mean、Std，並計算 (A−Mean)/Std ==========
    // 語法: Eigen::Map<Eigen::MatrixXf> mat(指標, 行數, 列數);
    //       mat.setRandom();

    /* 請初始化矩陣為隨機值 */


    // ========== TODO 3: 利用 std::chrono 記錄整體執行時間 ==========
    auto start = std::chrono::high_resolution_clock::now();


    // ========== TODO 4: 執行 Normalization Kernel ==========
    // 每一列分配一個 Block
    // 語法: normalize_kernel<<<rows, 1>>>(d_data, rows, cols);

    /* 請執行 Kernel */

    cudaDeviceSynchronize();


    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    // ========== 輸出結果 ==========
    std::cout << "Normalization 時間: " << diff.count() << " s" << std::endl;


    // ========== 清理資源 ==========
    cudaFree(d_data);
    
    return 0;
}
