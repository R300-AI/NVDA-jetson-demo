#include <iostream>     // 用於 cout 輸出
#include <vector>       // 用於 vector 容器
#include <random>       // 用於產生隨機數
#include <chrono>       // 用於計時
#include <algorithm>    // 用於 std::generate

using namespace std;

int main() {

    cout << "\n【實驗提示】" << endl;
    cout << "請在執行前開啟 tegrastats，觀察:" << endl;
    cout << "1. CPU 頻率是否達到最高" << endl;
    cout << "2. VDD_CPU 功耗數值" << endl;

    // ========== TODO 1: 設定向量大小 ==========
    // 透過 "const int N = 數值;" 設定向量的大小

    const int N = 100;      /* 請填入正確的向量空間的大小 */
    
    
    // ========== TODO 2: 建立向量 ==========
    // 透過 "vector<float> 變數名(大小)" 建立一個能夠容納 N 個 float 型態資料的A, B, C向量空間

    vector<float> A(N);          /* 請接著再創建B, C向量空間 */

    // ========== TODO 3: 填充隨機數值 ==========
    // 透過 "generate()" 函數搭配隨機數產生器填充向量 A 和 B
    
    std::random_device rd;                                       // 建立隨機種子
    std::mt19937 generator(rd());                                // 建立隨機數生成引擎
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);      // 建立[0.0, 1.0]均勻分布的配置
    auto generate_random_number = [&]() { return dist(generator); };    // 建立 Lambda 函數

    std::generate(A.begin(), A.end(), generate_random_number);   /* 請接著再為B向量空間填充隨機數 */

    // ========== 開始計時 ==========
    // 透過 "chrono::high_resolution_clock::now()" 取得目前時間點戳記
    auto start = chrono::high_resolution_clock::now(); 

    // ========== TODO 4: 向量加法 (使用 for-loop) ==========
    for(int i = 0; i < N; ++i) {
        /* 請修正下列程式碼，以完成個別元素的加法運算 */
        C[i] = A[i];
    }

    // ========== 結束計時 ==========
    // 透過 "chrono::high_resolution_clock::now()" 取得目前時間點戳記，並計算經過的時間差
    auto end = chrono::high_resolution_clock::now();      
    chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 ==========
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    cout << "向量大小: " << N << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl; 
    
    return 0;
}
