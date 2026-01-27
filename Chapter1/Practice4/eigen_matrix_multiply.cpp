#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>    // 用於 std::generate
#include <cblas.h>

using namespace std;
using namespace Eigen;

extern "C" {
    void openblas_set_num_threads(int num_threads);
}

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
    
    // ========== TODO 1: 建立 Eigen 矩陣 ==========
    // 透過 "MatrixXf 變數名(行, 列)" 建立 Eigen 矩陣
    MatrixXf A(N, N);             // 範例：建立 A
    // MatrixXf B(N, N);         /* 請接著建立 B 矩陣 */
    // MatrixXf C_eigen(N, N);   /* 請接著建立 C_eigen 矩陣（Eigen 結果）*/

    // ========== TODO 2: 填充隨機數值 ==========
    // 透過 "fill_random_uniform_0_1()" 填充矩陣 A 和 B
    fill_random_uniform_0_1(A);   // 範例：填充 A
    // fill_random_uniform_0_1(B);   /* 請接著填充 B 矩陣 */

    // ========== TODO 3: Eigen 矩陣乘法測試 ==========
    auto start_eigen = chrono::high_resolution_clock::now();
    // C_eigen = A * B;          /* 請使用 Eigen 完成矩陣乘法 */
    auto end_eigen = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_eigen = end_eigen - start_eigen;
    cout << "Eigen 執行時間: " << elapsed_eigen.count() << " 秒" << endl;

    // ========== TODO 4: OpenBLAS 單核對比測試 ==========
    // 將 A, B 轉成一維陣列供 OpenBLAS 使用
    vector<float> A_data(N*N);    // 範例：A 轉一維
    // vector<float> B_data(N*N);/* 請接著建立 B_data */
    // vector<float> C_data(N*N);/* 請接著建立 C_data */
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            A_data[i*N + j] = A(i, j); // 範例：A 轉一維
            // B_data[i*N + j] = B(i, j); /* 請接著轉換 B 矩陣 */
        }
    }

    // 設定 OpenBLAS 為單執行緒
    openblas_set_num_threads(1);
    auto start_blas = chrono::high_resolution_clock::now();
    // 呼叫 cblas_sgemm 完成矩陣乘法 C = A × B
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             N, N, N, 1.0f, A_data.data(), N, B_data.data(), N, 0.0f, C_data.data(), N);  /* 請完成單核 OpenBLAS 呼叫 */
    auto end_blas = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_blas = end_blas - start_blas;
    cout << "OpenBLAS (單核) 執行時間: " << elapsed_blas.count() << " 秒" << endl;
    cout << "Eigen vs OpenBLAS 速度比: " << elapsed_eigen.count() / elapsed_blas.count() << endl;
    
    return 0;
}
