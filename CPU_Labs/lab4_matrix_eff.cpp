#include <iostream>
#include <vector>
#include <chrono>
#include <cblas.h>
#include <Eigen/Dense>

using namespace Eigen;

void run_test(int threads, int N, const float* ptr_A, const float* ptr_B, float* ptr_C) {
    // 2. 設定執行緒
    openblas_set_num_threads(threads);

    std::cout << ">>> Threads: " << threads << std::endl;
    std::cout << "    按下 Enter 開始運算 (請監測 VDD_CPU)...";
    std::cin.get(); 

    auto start = std::chrono::high_resolution_clock::now();

    // 3. 執行 cblas_sgemm
    // C = 1.0 * A * B + 0.0 * C
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, ptr_A, N, ptr_B, N, 0.0f, ptr_C, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "    執行時間: " << diff.count() << " s" << std::endl << std::endl;
}

int main() {
    // 題目要求: 2048 x 2048
    const int N = 2048;
    std::cout << "初始化矩陣 (2048 x 2048)..." << std::endl;

    // 1. 使用 Eigen 建立隨機矩陣 (這比 std::vector 方便)
    MatrixXf MatA = MatrixXf::Random(N, N);
    MatrixXf MatB = MatrixXf::Random(N, N);
    MatrixXf MatC(N, N); // 結果矩陣

    std::vector<int> thread_list = {1, 2, 4, 8};

    for(int t : thread_list) {
        // 傳入資料指標 (data()) 給 OpenBLAS 使用
        run_test(t, N, MatA.data(), MatB.data(), MatC.data());
    }

    return 0;
}
