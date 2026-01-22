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
    
    
    // ========== TODO 2: 建立 Eigen 向量 ==========
    // VectorXf 是 Eigen 的單精度浮點數向量型別
    // 
    // 【命名規則】
    //   VectorXf
    //   ├─ Vector: 向量
    //   ├─ X: 動態大小（eXtendable），在執行時決定大小
    //   │   如果是固定大小，會用數字，例如 Vector3f (3個元素)
    //   └─ f: float 型別（單精度浮點數）
    //       如果是 double，會是 VectorXd
    //
    // 【語法詳解】
    //   VectorXf 變數名(大小);
    //   └─ (大小): 圓括號內是建構函數的參數，指定向量元素數量
    //
    VectorXf A(N);      /* 請接著再創建B, C向量 */
    
    
    
    
    // ========== TODO 3: 填充隨機數值 ==========
    // setRandom() 是 Eigen 的內建成員函數
    // 
    // 【語法詳解】
    //   A.setRandom();
    //   ├─ A: Eigen 向量物件
    //   ├─ .: 成員存取運算子
    //   └─ setRandom(): 成員函數，自動填充 [-1, 1] 之間的隨機數
    //
    // 對比 Practice1:
    //   Practice1 需要 4 行程式碼 (rd, gen, dis, generate)
    //   Practice2 只需要 1 行 (setRandom)
    
    A.setRandom();      /* 請接著為B向量填充隨機數 */
    
    
    
    
    // ========== TODO 4: 開始計時 ==========
    // 與 Practice1 相同的計時方式
    
    auto start = chrono::high_resolution_clock::now();      /* 請建立開始時間點 */
    
    
    // ========== TODO 5: Eigen 向量加法 ==========
    // Eigen 支援運算子重載 (operator overloading)
    // 
    // 【語法詳解】
    //   C = A + B;
    //   ├─ C, A, B: 都是 VectorXf 物件
    //   ├─ +: 加號被重載，不是單純的數值相加
    //   └─ =: 賦值運算子，將結果存入 C
    //
    // 【重要概念】運算子重載
    //   在 C++ 中，+ 號可以被「重新定義」
    //   - 對於 int, float: + 是數值相加
    //   - 對於 VectorXf: + 是向量逐元素相加（使用 SIMD 優化）
    //
    // 內部運作:
    //   Eigen 會自動使用 ARM NEON 指令集
    //   一次處理多個元素（例如一次處理 4 個 float）
    //   因此比 for-loop 快很多
    
    C = A + B;      /* 請使用 Eigen 運算子完成向量加法 */
    
    
    // ========== TODO 6: 結束計時 ==========
    // 取得結束時間並計算時間差
    // 
    // 【語法提醒】
    //   auto end = chrono::high_resolution_clock::now();
    //   chrono::duration<double> elapsed = end - start;
    
    auto end = chrono::high_resolution_clock::now();      /* 請建立結束時間點 */
    chrono::duration<double> elapsed = end - start;      /* 請計算時間差 */
    
    
    // ========== TODO 7: 輸出結果 ==========
    // 【語法提醒】
    //   cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    //   └─ elapsed.count() 回傳 double 型別的秒數
    
    cout << "向量大小: " << N << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    
    cout << "\n【實驗重點】" << endl;
    cout << "比較 Practice1 (for-loop) 與 Practice2 (Eigen) 的執行時間:" << endl;
    cout << "- Eigen 使用了 SIMD 指令集 (如 ARM NEON)" << endl;
    cout << "- 一次處理多個資料，因此速度更快" << endl;
    cout << "- 同時觀察 tegrastats 的 CPU 使用率變化" << endl;
    
    return 0;
}
