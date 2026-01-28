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

    cout << "\n【實驗提示】" << endl;
    cout << "請提前開啟 tegrastats，觀察:" << endl;
    cout << "1. CPU 頻率" << endl;
    cout << "2. VDD_CPU 功耗數值" << endl;

    // ========== TODO 1: 設定矩陣大小 ==========
    // 透過 "const int N = 數值;" 設定矩陣的大小

    const int N = 1024;                /* 請填入正確的矩陣大小 */
    
    
    // ========== TODO 2: 建立矩陣資料 ==========
    // 透過 "vector<float> 變數名(大小)" 建立一維陣列儲存矩陣 (大小為 N * N)

    vector<float> A(N * N);            /* 請接著建立 B, C 矩陣 */


    // ========== TODO 3: 填充隨機數值 ==========
    // 透過 "fill_random_uniform_0_1()" 填充矩陣 A 和 B

    fill_random_uniform_0_1(A);        /* 請接著填充 B 矩陣 */


    // ========== 開始計時 ==========
    auto start = chrono::high_resolution_clock::now();

    
    // ========== TODO 4: 矩陣乘法 (使用 OpenBLAS) ==========
    // cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    // C = alpha * A * B + beta * C

    int threads = 1;                   /* 請設定執行緒數量 */
    openblas_set_num_threads(threads);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 N, N, N, 1.0f, A.data(), N, A.data(), N, 0.0f, A.data(), N);  /* 請修正矩陣參數 */  
    
    
    // ========== 結束計時 ==========
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;


    // ========== 輸出結果 =========
    cout << "矩陣大小: " << N << " x " << N << endl;
    cout << "執行緒數量: " << threads << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    return 0;
}
