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
    Eigen::MatrixXf A(N, N);             // 範例：建立 A
    // MatrixXf B(N, N);         /* 請接著建立 B 矩陣 */
    // MatrixXf C(N, N);   /* 請接著建立 C 矩陣（Eigen 結果）*/

    // ========== TODO 2: 填充隨機數值 ==========
    // 透過 "fill_random_uniform_0_1()" 填充矩陣 A 和 B
    fill_random_uniform_0_1(A);   // 範例：填充 A
    // fill_random_uniform_0_1(B);   /* 請接著填充 B 矩陣 */


    // ========== 開始計時 ==========
    auto start = chrono::high_resolution_clock::now();

    
    // ========== TODO 4: 矩陣乘法 (使用 Eigen) ==========
    // C = A * B;          /* 請使用 Eigen 完成矩陣乘法 */


    // ========== 結束計時 ==========
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;


    // ========== 輸出結果 =========
    cout << "矩陣大小: " << N << " x " << N << endl;
    cout << "Eigen 執行時間: " << elapsed_eigen.count() << " 秒" << endl;

    return 0;
}
