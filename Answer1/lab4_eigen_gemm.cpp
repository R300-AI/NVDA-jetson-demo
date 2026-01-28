#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <chrono>

// 填充隨機數 [0, 1)
template<typename VectorType>
static void fill_random_uniform_0_1(VectorType& vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec(i) = (float)std::rand() / RAND_MAX;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats 觀察 CPU 頻率與功耗" << std::endl;

    // ========== 設定矩陣大小 ==========
    const int N = 2048;
    
    std::cout << "\n矩陣大小: " << N << " x " << N << std::endl;

    // ========== 建立 Eigen 矩陣 ==========
    Eigen::MatrixXf A(N, N);
    Eigen::MatrixXf B(N, N);
    Eigen::MatrixXf C(N, N);

    // ========== 填充隨機數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A);
    fill_random_uniform_0_1(B);

    // ========== 開始計時 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // ========== 矩陣乘法 (使用 Eigen) ==========
    C = A * B;

    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 ==========
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;
    std::cout << "執行時間: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}
