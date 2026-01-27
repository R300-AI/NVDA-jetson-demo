#include <iostream>     // 用於 cout 輸出
#include <Eigen/Dense>  // Eigen 線性代數函式庫
#include <random>       // 用於產生隨機數
#include <chrono>       // 用於計時
#include <algorithm>    // 用於 std::generate

using namespace std;
using namespace Eigen;  // 使用 Eigen 命名空間

static void fill_random_uniform_0_1(float* data, size_t count) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto gen = [&]() { return dist(generator); };
    std::generate(data, data + count, gen);
}

int main() {

    cout << "\n【實驗提示】" << endl;
    cout << "請在執行前開啟 tegrastats，觀察:" << endl;
    cout << "1. CPU 頻率" << endl;
    cout << "2. VDD_CPU 功耗數值" << endl;

    // ========== TODO 1: 設定向量大小 ==========
    // 透過 "const int N = 數值;" 設定向量的大小

    const int N = 100;            /* 請填入正確的向量空間的大小 */
    
    
    // ========== TODO 2: 建立向量 ========== 
    // 透過 "vector<float> 變數名(大小)" 建立一個能夠容納 N 個 float 型態資料的A, B, C向量空間
    Eigen::VectorXf<float> A(N);           // 範例：建立 A


    // ========== TODO 3: 填充隨機數值 ==========
    // 透過 "fill_random_uniform_0_1()" 填充向量 A 和 B
    fill_random_uniform_0_1(A.data(), static_cast<size_t>(A.size())); // 範例：填充 A


    // ========== 開始計時 ==========
    auto start = chrono::high_resolution_clock::now();

    
    // ========== TODO 4: 向量加法 (使用 Eigen) ==========
    // C = A + B
    // C = A + B;      
    A = A.array() + A.array();     /* 請依上方要求完成向量加法 */


    // ========== 結束計時 ==========
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    
    // ========== 輸出結果 =========
    cout << "向量大小: " << N << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    return 0;
}
