#include <iostream>     // 用於 cout 輸出
#include <vector>       // 用於 vector 容器
#include <random>       // 用於產生隨機數
#include <chrono>       // 用於計時
#include <cblas.h>      // OpenBLAS 的 C 介面

// 宣告 OpenBLAS 執行緒控制函數
extern "C" {
    void openblas_set_num_threads(int num_threads);
}

using namespace std;

int main() {
    const int N = 100000000;      // 範例：設定 N
    
    // ========== TODO 1: 建立向量 ==========
    vector<float> A(N);           // 範例：建立 A
    // vector<float> B(N);       /* 請接著建立 B 向量 */
    // vector<float> C(N);       /* 請接著建立 C 向量 */

    // ========== TODO 2: 填充隨機數值 ==========
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto generate_random_number = [&]() { return dist(generator); };
    std::generate(A.begin(), A.end(), generate_random_number);   // 範例：填充 A
    // std::generate(B.begin(), B.end(), generate_random_number);   /* 請接著填充 B 向量 */

    // ========== TODO 3: 多執行緒測試 ==========
    // 設定執行緒數量（範例：1）
    int threads = 1;
    openblas_set_num_threads(threads);
    auto start = chrono::high_resolution_clock::now();
    // vector<float> C = B;   /* 請接著建立 C 並複製 B */
    // cblas_saxpy(N, 1.0f, A.data(), 1, C.data(), 1);   /* 請呼叫 cblas_saxpy 完成加法 */
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "執行緒數量: " << threads << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    // threads = ...; openblas_set_num_threads(threads); /* 請自行修改 threads 進行多執行緒測試 */
    return 0;
}
