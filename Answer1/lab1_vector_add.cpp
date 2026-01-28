#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

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

    // ========== 設定向量大小 ==========
    const int N = 10000000;  // 10^7
    
    std::cout << "\n向量大小: " << N << " (10^7)" << std::endl;

    // ========== 建立向量 ==========
    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // ========== 填充隨機數值 ==========
    std::srand(42);
    fill_random_uniform_0_1(A);
    fill_random_uniform_0_1(B);

    // ========== 開始計時 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // ========== 向量加法 (使用 for-loop) ==========
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 ==========
    std::cout << "向量大小: " << N << std::endl;
    std::cout << "執行時間: " << elapsed.count() << " 秒" << std::endl;
    
    return 0;
}
