#include <iostream>     // 用於 cout 輸出
#include <vector>       // 用於 vector 容器
#include <random>       // 用於產生隨機數
#include <chr        // ========== TODO 7: 結束計時並輸出 ==========
        
        auto end = chrono::high_resolution_clock::now();      /* 請建立結束時間點 */
        chrono::duration<double> elapsed = end - start;      /* 請計算時間差 */
        
        cout << "執行緒數量: " << threads << endl;
        cout << "執行時間: " << elapsed.count() << " 秒" << endl;
        cout << "-------------------" << endl;
    }   // 用於計時
#include <cblas.h>      // OpenBLAS 的 C 介面

// 宣告 OpenBLAS 執行緒控制函數
extern "C" {
    void openblas_set_num_threads(int num_threads);
}

using namespace std;

int main() {

    cout << "\n【實驗提示】" << endl;
    cout << "請在執行前開啟 tegrastats，觀察:" << endl;
    cout << "1. 多核心 CPU 使用率變化" << endl;
    cout << "2. VDD_CPU 功耗隨執行緒數量的變化" << endl;

    const int N = 100000000;  // 10^8
    
    // ========== TODO 1: 建立向量 A 和 B ==========
    // 參考 Practice1 的做法: vector<float> 變數名(大小);
    
    vector<float> A(N);      /* 請接著創建B向量 */
    
    
    
    
    // ========== TODO 2: 填充隨機數值 ==========
    // 參考 Practice1 的 random_device, mt19937, uniform_real_distribution
    // 使用 generate() 填充 A 和 B
    
    std::random_device rd;                                       // 建立隨機種子
    std::mt19937 generator(rd());                                // 建立隨機數生成引擎
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);      // 建立[0.0, 1.0]均勻分布的配置
    auto generate_random_number = [&]() { return dist(generator); };    // 建立 Lambda 函數

    std::generate(A.begin(), A.end(), generate_random_number);   /* 請接著再為B向量空間填充隨機數 */
    
    
    
    
    
    
    // ========== TODO 3: 測試不同執行緒數量 ==========
    // 建立陣列儲存要測試的執行緒數: 1, 2, 4, 8
    // 使用 range-based for loop 遍歷
    // 
    // 【語法詳解 1】陣列初始化
    //   int thread_counts[] = {1, 2, 4, 8};
    //   ├─ int: 整數型別
    //   ├─ thread_counts: 陣列名稱
    //   ├─ []: 中括號表示陣列（大小會自動推斷）
    //   └─ {1, 2, 4, 8}: 大括號內是初始值列表
    //
    // 【語法詳解 2】Range-based for loop (C++11)
    //   for(int threads : thread_counts) {
    //       // 迴圈本體
    //   }
    //   ├─ int threads: 迴圈變數，每次迭代會取得陣列的一個元素
    //   ├─ :: 冒號表示「在...之中」
    //   └─ thread_counts: 要遍歷的陣列
    //
    // 類比 Python:
    //   thread_counts = [1, 2, 4, 8]
    //   for threads in thread_counts:
    //       # 迴圈本體
    
    int thread_counts[] = {1, 2, 4, 8};      /* 請建立要測試的執行緒數量陣列 */
    
    for(int threads : thread_counts) {      /* 請使用 range-based for loop 遍歷 */
        // ========== TODO 4: 設定執行緒數量 ==========
        // openblas_set_num_threads(n) 會告訴 OpenBLAS 使用 n 個執行緒
        // 
        // 【語法詳解】
        //   openblas_set_num_threads(threads);
        //   ├─ openblas_set_num_threads: 函數名稱
        //   ├─ (threads): 圓括號內是參數，要使用的執行緒數量
        //   └─ ;: 分號結尾（C++ 的語句結束符號）
        //
        // 這是多核心平行運算的關鍵設定
        // 例如: threads=4 表示使用 4 個 CPU 核心同時運算
        
        openblas_set_num_threads(threads);      /* 請設定執行緒數量 */
        
        
        // ========== TODO 5: 開始計時 ==========
        
        auto start = chrono::high_resolution_clock::now();      /* 請建立開始時間點 */
        
        // ========== TODO 6: 使用 cblas_saxpy 完成向量加法 ==========
        // cblas_saxpy 的功能: Y = alpha * X + Y
        // 
        // 【函數名稱解析】
        //   cblas_saxpy
        //   ├─ cblas: C 語言的 BLAS 介面
        //   ├─ s: single precision (單精度，float)
        //   ├─ axpy: alpha*X plus Y (數學運算名稱)
        //
        // 【重要觀念】為什麼要複製 B？
        //   因為 cblas_saxpy 會「修改」第二個向量
        //   如果直接用 B，B 的值會被改變
        //   所以要先建立 C 的副本，讓 C = B
        //
        // 【語法詳解】
        //   vector<float> C = B;
        //   └─ 複製建構函數，建立 B 的副本並命名為 C
        //
        //   cblas_saxpy(N, 1.0f, A.data(), 1, C.data(), 1);
        //   ├─ N: 向量長度（元素數量）
        //   ├─ 1.0f: alpha 係數（f 表示 float）
        //   │   設為 1.0 表示 Y = 1.0*X + Y，也就是 C = A + C
        //   ├─ A.data(): 取得向量 A 的記憶體位址（指標）
        //   │   .data() 是 vector 的成員函數，回傳底層陣列的指標
        //   ├─ 1: X 的步進 (stride)
        //   │   1 表示連續存取（每次跳 1 個元素）
        //   ├─ C.data(): 取得向量 C 的記憶體位址
        //   └─ 1: Y 的步進
        //
        // 執行流程:
        //   1. C = B (複製)
        //   2. C = 1.0*A + C  →  C = A + B ✓
        
        vector<float> C = B;      /* 請先建立C並複製B的內容 */
        cblas_saxpy(N, 1.0f, A.data(), 1, C.data(), 1);      /* 請呼叫 cblas_saxpy 完成加法 */
        
        
        // ========== TODO 7: 結束計時並輸出 ==========
        
        
        
        cout << "執行緒數量: " << "___" << endl;
        cout << "執行時間: " << "___" << " 秒" << endl;
        cout << "-------------------" << endl;
    
    
    cout << "\n【實驗重點】" << endl;
    cout << "觀察多核心的效能提升:" << endl;
    cout << "1. 執行緒從 1 增加到 2, 4, 8 時的執行時間變化" << endl;
    cout << "2. 是否呈現線性加速？(2 倍執行緒 = 2 倍速度？)" << endl;
    cout << "3. 使用 tegrastats 觀察多個 CPU 核心的使用率" << endl;
    
    return 0;
}
