#include <iostream>     // 用於 cout 輸出
#include <Eigen/Dense>  // Eigen 線性代數函式庫
#include <chrono>       // 用於計時
#include <cblas.h>      // OpenBLAS 的 C 介面

// 宣告 OpenBLAS 執行緒控制函數
extern "C" {
    void openblas_set_num_threads(int num_threads);
}

using namespace std;
using namespace Eigen;

int main() {

    cout << "\n【實驗提示】" << endl;
    cout << "請在執行前開啟 tegrastats，觀察:" << endl;
    cout << "1. 多核心 CPU 使用率變化" << endl;
    cout << "2. VDD_CPU 功耗隨執行緒數量的變化" << endl;
    cout << "3. 找出效率最高的執行緒數量(甜蜜點)" << endl;

    const int N = 2048;  // 矩陣大小 2048 x 2048
    
    // ========== TODO 1: 建立 Eigen 矩陣 ==========
    // MatrixXf 是 Eigen 的單精度浮點數矩陣型別
    // 語法: MatrixXf 變數名(列數, 行數);
    
    MatrixXf A(N, N);      /* 請接著創建B矩陣 */
    
    
    
    
    // ========== TODO 2: 填充隨機數值 ==========
    // 使用 setRandom() 填充矩陣
    
    A.setRandom();      /* 請接著為B矩陣填充隨機數 */
    
    
    // ========== TODO 3: 準備 OpenBLAS 使用的資料 ==========
    // OpenBLAS 使用 C 風格的一維陣列
    // 需要將 Eigen 的二維矩陣轉換成一維 vector
    // 語法: vector<float> A_data(N*N), B_data(N*N), C_data(N*N);
    
    vector<float> A_data(N*N);      /* 請接著創建B_data和C_data向量 */
    
    
    
    
    // ========== TODO 4: 資料格式轉換 (Eigen -> vector) ==========
    // Eigen 預設使用 column-major (列優先)
    // OpenBLAS 這裡使用 row-major (行優先)
    // 需要用雙層迴圈轉換
    //
    // 【重要概念】Row-major vs Column-major
    //   以 2x2 矩陣為例: [[1, 2], [3, 4]]
    //   
    //   Row-major (行優先): 按「列」儲存
    //   記憶體順序: [1, 2, 3, 4]
    //   ├─ 先存第 0 列: 1, 2
    //   └─ 再存第 1 列: 3, 4
    //
    //   Column-major (列優先): 按「行」儲存
    //   記憶體順序: [1, 3, 2, 4]
    //   ├─ 先存第 0 行: 1, 3
    //   └─ 再存第 1 行: 2, 4
    //
    // 【語法詳解】二維索引轉一維索引
    //   A_data[i*N + j] = A(i, j);
    //   ├─ A(i, j): Eigen 矩陣存取，第 i 列第 j 行
    //   │   圓括號 () 是 Eigen 的運算子重載
    //   ├─ i*N + j: 二維轉一維的索引公式
    //   │   ├─ i: 列索引 (0 到 N-1)
    //   │   ├─ j: 行索引 (0 到 N-1)
    //   │   └─ i*N + j: 計算在一維陣列中的位置
    //   └─ A_data[...]: vector 陣列存取
    //       中括號 [] 是陣列下標運算子
    //
    // 範例: N=3, 要存取 (1,2) 位置 (第1列第2行)
    //   索引 = 1*3 + 2 = 5
    //   意思: 跳過第0列的3個元素，再跳過第1列的前2個元素
    
    for(int i = 0; i < N; ++i) {        /* 請建立雙層迴圈遍歷矩陣 */
        for(int j = 0; j < N; ++j) {
            A_data[i*N + j] = A(i, j);  /* 請接著轉換B矩陣 */
        }
    }
    
    
    
    
    
    
    
    // 測試不同執行緒數量
    int thread_counts[] = {1, 2, 4, 8};
    
    cout << "矩陣大小: " << N << " x " << N << endl;
    cout << "====================================" << endl;
    
    for(int threads : thread_counts) {
        
        // ========== TODO 5: 設定執行緒數量 ==========
        // 使用 openblas_set_num_threads(threads);
        
        openblas_set_num_threads(threads);      /* 請設定執行緒數量 */
        
        
        // ========== TODO 6: 開始計時 ==========
        
        auto start = chrono::high_resolution_clock::now();      /* 請建立開始時間點 */
        
        // ========== TODO 7: 矩陣乘法 (cblas_sgemm) ==========
        // cblas_sgemm 是通用矩陣乘法 (General Matrix Multiply)
        // 功能: C = alpha * A * B + beta * C
        //
        // 【函數名稱解析】
        //   cblas_sgemm
        //   ├─ cblas: C 語言的 BLAS 介面
        //   ├─ s: single precision (單精度，float)
        //   ├─ gemm: GEneral Matrix Multiply (通用矩陣乘法)
        //
        // 【數學公式】
        //   C = alpha * op(A) * op(B) + beta * C
        //   ├─ alpha, beta: 係數
        //   ├─ op(A): A 或 A 的轉置
        //   └─ op(B): B 或 B 的轉置
        //
        // 【參數詳解】
        //   cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
        //
        //   1. order: 儲存順序
        //      CblasRowMajor: 行優先 (C/C++ 風格)
        //      CblasColMajor: 列優先 (Fortran 風格)
        //
        //   2. transA, transB: 是否轉置
        //      CblasNoTrans: 不轉置
        //      CblasTrans: 轉置
        //
        //   3. M, N, K: 矩陣維度
        //      C[M×N] = A[M×K] * B[K×N]
        //      對於正方形矩陣: M = N = K = 矩陣大小
        //
        //   4. alpha: 係數 (我們設為 1.0f)
        //
        //   5. A, lda:
        //      A: A 矩陣的指標 (A_data.data())
        //      lda: leading dimension of A (A 的首維度)
        //          對於行優先，lda = A 的行數
        //
        //   6. B, ldb: 同上，B 矩陣
        //
        //   7. beta: 係數 (我們設為 0.0f，表示不保留 C 的舊值)
        //
        //   8. C, ldc: 同上，C 矩陣（輸出）
        //
        // 【.data() 說明】
        //   A_data.data()
        //   ├─ A_data: vector<float> 物件
        //   ├─ .: 成員存取運算子
        //   └─ data(): 成員函數，回傳底層陣列的指標
        //       這是因為 C 函數需要原始指標，不能直接用 vector
        
        cblas_sgemm(CblasRowMajor,      /* 請完成 cblas_sgemm 呼叫 */
                    CblasNoTrans,       /* 提示: 參數依序為 order, transA, transB, */
                    CblasNoTrans,       /*      M, N, K, alpha, A, lda, */
                    N, N, N,            /*      B, ldb, beta, C, ldc */
                    1.0f,
                    A_data.data(), N,
                    B_data.data(), N,
                    0.0f,
                    C_data.data(), N);
        
        
        // ========== TODO 8: 結束計時並輸出 ==========
        
        auto end = chrono::high_resolution_clock::now();      /* 請建立結束時間點 */
        chrono::duration<double> elapsed = end - start;      /* 請計算時間差 */
        
        cout << "執行緒數量: " << threads << endl;
        cout << "執行時間: " << elapsed.count() << " 秒" << endl;
        cout << "-----------------------------------" << endl;
    }
    
    cout << "\n【作業任務】" << endl;
    cout << "1. 執行程式時，請同時開啟 tegrastats 監測 VDD_CPU 功耗" << endl;
    cout << "   指令: sudo tegrastats" << endl;
    cout << "\n2. 記錄各執行緒數量的數據:" << endl;
    cout << "   執行緒 | 執行時間(秒) | VDD_CPU(mW)" << endl;
    cout << "   -------|-------------|------------" << endl;
    cout << "      1   |             |           " << endl;
    cout << "      2   |             |           " << endl;
    cout << "      4   |             |           " << endl;
    cout << "      8   |             |           " << endl;
    cout << "\n3. 計算效率 (Efficiency):" << endl;
    cout << "   Efficiency = 1 / (執行時間 × VDD_CPU)" << endl;
    cout << "   數值越大代表效率越高" << endl;
    cout << "\n4. 找出效率最高的執行緒數量 (甜蜜點)" << endl;
    cout << "   思考: 為什麼不是執行緒越多越好？" << endl;
    
    return 0;
}
