#include <iostream>     // 用於 cout 輸出
#include <Eigen/Dense>  // Eigen 線性代數函式庫
#include <chrono>       // 用於計時

using namespace std;
using namespace Eigen;  // 使用 Eigen 命名空間

int main() {

    cout << "\n【實驗提示】" << endl;
    cout << "請在執行前開啟 tegrastats，觀察:" << endl;
    cout << "1. CPU 頻率是否達到最高" << endl;
    cout << "2. VDD_CPU 功耗數值" << endl;

    // ========== TODO 1: 設定向量大小 ==========
    // 與 Practice1 相同，使用 10^8 大小
    
    const int N = 100000000;      /* 請填入正確的向量大小 */
    
    
    // ========== TODO 2: 建立向量 ==========
    // A, B, C 都要建立
    VectorXf A(N);      // 範例：建立 A
    // VectorXf B(N);   /* 請接著再創建B向量 */
    // VectorXf C(N);   /* 請接著再創建C向量 */

    // ========== TODO 3: 填充隨機數值 ==========
    // 對 A, B 填充隨機數
    A.setRandom();      // 範例：填充 A
    // B.setRandom();   /* 請接著為B向量填充隨機數 */

    // ========== TODO 4: 向量加法 (使用 Eigen) ==========
    // C = A + B
    // C = A + B;      /* 請依上方要求完成向量加法 */

    // ========== 開始計時 ==========
    auto start = chrono::high_resolution_clock::now();

    // ========== 結束計時 ==========
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // ========== 輸出結果 =========
    cout << "向量大小: " << N << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    return 0;
}
