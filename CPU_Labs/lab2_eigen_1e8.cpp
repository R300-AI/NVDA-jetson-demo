#include <iostream>
#include <chrono>
#include <omp.h>
#include <Eigen/Dense>

using namespace Eigen;

int main() {
    // 題目要求: 10^8 (一億)
    const long size = 100000000;
    
    // 計算記憶體用量: 10^8 * 4 bytes * 3 (A, B, C) = 1.2 GB
    std::cout << "向量大小: " << size << " (預計占用約 1.2 GB RAM)" << std::endl;
    std::cout << "正在初始化隨機向量 (請耐心等待)..." << std::endl;

    // 1. 使用 Eigen::VectorXf 建立向量
    // setRandom() 會使用亂數填滿，這步驟在 CPU 上會花一點時間
    VectorXf A(size);
    VectorXf B(size);
    VectorXf C(size);

    A.setRandom();
    B.setRandom();

    std::cout << "初始化完成，準備進行效能測試。" << std::endl;

    // --- Part 1: Eigen 多核加法 (Simd + OpenMP) ---
    // 設定使用 4 核心 (可依需求調整)
    Eigen::setNbThreads(4);
    
    auto start_eigen = std::chrono::high_resolution_clock::now();
    
    // 2. 直接使用 C = A + B
    C = A + B;
    
    auto end_eigen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_eigen = end_eigen - start_eigen;
    
    std::cout << "[Eigen Multicore] Time: " << diff_eigen.count() << " s" << std::endl;


    // --- Part 2: C++ For-loop 單核加法 (對照組) ---
    // 為了公平比較，我們手寫一個單核迴圈
    // 注意: 為了避免快取 (Cache) 影響，這裡再次執行一次
    auto start_loop = std::chrono::high_resolution_clock::now();
    
    for (long i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
    
    auto end_loop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_loop = end_loop - start_loop;
    
    std::cout << "[C++ Loop Single] Time: " << diff_loop.count() << " s" << std::endl;
    
    // 計算加速倍率
    std::cout << "加速倍率: " << diff_loop.count() / diff_eigen.count() << "x" << std::endl;

    return 0;
}
