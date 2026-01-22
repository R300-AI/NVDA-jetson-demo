#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <chrono>
#include <cblas.h>

using namespace std;
using namespace Eigen;

extern "C" {
    void openblas_set_num_threads(int num_threads);
}

int main() {
    const int N = 2048;           // 範例：設定 N
    
    // ========== TODO 1: 建立矩陣 ==========
    // 建立 A, B, C 三個 N x N 矩陣
    MatrixXf A(N, N);             // 範例：建立 A
    // MatrixXf B(N, N);         /* 請接著建立 B 矩陣 */
    // MatrixXf C(N, N);         /* 請接著建立 C 矩陣 */

    // ========== TODO 2: 填充隨機數值 ==========
    // 對 A, B 填充隨機數
    A.setRandom();                // 範例：填充 A
    // B.setRandom();            /* 請接著填充 B 矩陣 */

    // ========== TODO 3: 準備 BLAS 資料 ==========
    // 將 A, B, C 轉成一維陣列
    vector<float> A_data(N*N);    // 範例：A 轉一維
    // vector<float> B_data(N*N);/* 請接著建立 B_data */
    // vector<float> C_data(N*N);/* 請接著建立 C_data */
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            A_data[i*N + j] = A(i, j); // 範例：A 轉一維
            // B_data[i*N + j] = B(i, j); /* 請接著轉換 B 矩陣 */
        }
    }

    // ========== TODO 4: 多執行緒測試 ==========
    // 設定執行緒數量（範例：1）
    int threads = 1;
    openblas_set_num_threads(threads);
    auto start = chrono::high_resolution_clock::now();
    // 呼叫 cblas_sgemm 完成矩陣乘法
    // cblas_sgemm(...)   /* 請依題目要求呼叫 cblas_sgemm 完成矩陣乘法 */
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "執行緒數量: " << threads << endl;
    cout << "執行時間: " << elapsed.count() << " 秒" << endl;
    // threads = ...; openblas_set_num_threads(threads); /* 請自行修改 threads 進行多執行緒測試 */
    return 0;
}
