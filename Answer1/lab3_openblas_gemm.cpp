#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cblas.h>

// 宣告 OpenBLAS 執行緒控制函數
extern "C" {
    void openblas_set_num_threads(int num_threads);
}

// 填充隨機數 [0, 1)
template<typename VectorType>
static void fill_random_uniform_0_1(VectorType& vec) {
    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats 觀察 CPU 頻率與功耗" << std::endl;

    // ========== 設定矩陣大小 ==========
    const int N = 2048;
    
    
    // ========== 建立矩陣資料 ==========
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);


    // ========== 填充隨機數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A);
    fill_random_uniform_0_1(B);


    // ========== 開始計時 ==========
    auto start = std::chrono::high_resolution_clock::now();

    
    // ========== 矩陣乘法 (使用 OpenBLAS) ==========
    int threads = 1;                   /* 請調整執行緒數量: 1, 2, 4, 8 */
    openblas_set_num_threads(threads);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A.data(), N, B.data(), N, 0.0f, C.data(), N);
    
    
    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    // ========== 輸出結果 =========
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;
    std::cout << "執行緒數量: " << threads << std::endl;
    std::cout << "執行時間: " << elapsed.count() << " 秒" << std::endl;
    
    return 0;
}
