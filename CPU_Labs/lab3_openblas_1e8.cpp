#include <iostream>
#include <vector>
#include <chrono>
#include <cblas.h>
#include <thread>

int main() {
    // 題目要求: 10^8
    const int n = 100000000;
    std::cout << "向量長度: " << n << std::endl;
    
    // 初始化 A 與 B
    // cblas_saxpy 計算的是 Y = alpha * X + Y
    // 所以我們把 A 當作 X，把 B 當作 Y (結果會存回 B)
    std::vector<float> A(n, 1.0f);
    std::vector<float> B(n, 2.0f); 
    
    // 備份一份 B，因為每次迴圈 B 都會被修改
    std::vector<float> B_backup = B;

    std::vector<int> thread_counts = {1, 2, 4, 8};

    std::cout << "--------------------------------------" << std::endl;

    for(int th : thread_counts) {
        // 1. 還原 B 向量 (為了讓每次運算量一致)
        B = B_backup;

        // 2. 設定 OpenBLAS 執行緒數量
        openblas_set_num_threads(th);

        std::cout << "Threads: " << th << " ... ";
        std::cout.flush();
        
        // 暫停一下讓 Tegrastats 更新 (監測 VDD_CPU)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto start = std::chrono::high_resolution_clock::now();

        // 3. 執行 cblas_saxpy (單精度 A*X + Y)
        // Y = 1.0 * A + B
        cblas_saxpy(n, 1.0f, A.data(), 1, B.data(), 1);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "Time: " << diff.count() << " s" << std::endl;
    }
    std::cout << "--------------------------------------" << std::endl;

    return 0;
}
