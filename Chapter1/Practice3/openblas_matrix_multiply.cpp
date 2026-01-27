#include <iostream>
#include <vector>
#include <random>
#include <chrono> 
#include <algorithm>    // 用於 std::generate
#include <cblas.h>      // OpenBLAS 的 C 介面

// 宣告 OpenBLAS 執行緒控制函數
extern "C" {
    void openblas_set_num_threads(int num_threads);
}

using namespace std;

template<typename VectorType>
static void fill_random_uniform_0_1(VectorType& vec) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto gen = [&]() { return dist(generator); };
    std::generate(vec.data(), vec.data() + vec.size(), gen);
}

int main() {
    const int N = 1024;           // 範例：設定矩陣大小 N x N
    
    // ========== TODO 1: 建立矩陣資料 ==========
    // 透過 "vector<float> 變數名(大小)" 建立一維陣列儲存矩陣
    vector<float> A(N * N);       // 範例：建立 A 矩陣（一維陣列）
    // vector<float> B(N * N);   /* 請接著建立 B 矩陣 */
    // vector<float> C(N * N);   /* 請接著建立 C 矩陣 */

    // ========== TODO 2: 填充隨機數值 ==========
    // 透過 "fill_random_uniform_0_1()" 填充矩陣 A 和 B
    fill_random_uniform_0_1(A);   // 範例：填充 A
    // fill_random_uniform_0_1(B);   /* 請接著填充 B 矩陣 */

    // ========== TODO 3: 多執行緒測試 ==========
    // 設定執行緒數量（範例：1）
    int threads = 1;
    openblas_set_num_threads(threads);
    auto start = chrono::high_resolution_clock::now();
    // 呼叫 cblas_sgemm 完成矩陣乘法 C = A × B
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             N, N, N, 1.0f, A.data(), N, B.data(), N, 0.0f, C.data(), N);  /* 請完成矩陣乘法呼叫 */
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "執行緒數量: " << threads << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    // threads = ...; openblas_set_num_threads(threads); /* 請自行修改 threads 進行多執行緒測試 */
    return 0;
}
