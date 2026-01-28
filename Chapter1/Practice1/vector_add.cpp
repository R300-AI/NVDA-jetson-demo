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

    // ========== TODO 1: 設定向量大小 ==========
    // 透過 "const int N = 數值;" 設定向量的大小

    const int N = 100;                 /* 請填入正確的向量空間的大小 */
    
    
    // ========== TODO 2: 建立向量 ========== 
    // 透過 "std::vector<float> 變數名(大小)" 建立向量空間

    std::vector<float> A(N);           /* 請接著建立 B, C 向量 */

    
    // ========== TODO 3: 填充隨機數值 ========== 
    // 透過 "fill_random_uniform_0_1()" 填充向量 A 和 B

    fill_random_uniform_0_1(A);        /* 請接著填充 B 向量 */


    // ========== 開始計時 ==========
    auto start = std::chrono::high_resolution_clock::now(); 


    // ========== TODO 4: 向量加法 (使用 for-loop) ==========
    // C = A + B
    for (int i = 0; i < N; ++i) {
         A[i] = A[i] + A[i];           /* 請依計算要求完成向量加法 */
    }


    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();      
    std::chrono::duration<double> elapsed = end - start;


    // ========== 輸出結果 =========
    std::cout << "向量大小: " << N << std::endl;
    std::cout << "執行時間: " << elapsed.count() << " 秒" << std::endl;
    
    return 0;
}
