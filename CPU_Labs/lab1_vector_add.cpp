#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

int main() {
    // 1. 初始化設定
    const size_t size = 100000000; // 10^8 元素 (約需 1.2GB RAM)
    std::cout << "正在初始化向量 (Size: " << size << ")..." << std::endl;

    std::vector<float> A(size);
    std::vector<float> B(size);
    std::vector<float> C(size);

    // 使用隨機數生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 利用 std::generate 填充數據
    auto gen_func = [&]() { return dis(gen); };
    std::generate(A.begin(), A.end(), gen_func);
    std::generate(B.begin(), B.end(), gen_func);

    std::cout << "開始計算 A + B ..." << std::endl;

    // 2. 記錄開始時間
    auto start = std::chrono::high_resolution_clock::now();

    // 3. 執行向量加法 (單核 For-loop)
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }

    // 4. 記錄結束時間
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "計算完成。" << std::endl;
    std::cout << "執行時間: " << diff.count() << " 秒" << std::endl;

    return 0;
}