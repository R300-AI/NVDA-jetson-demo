# NVIDIA Jetson CPU Optimization Labs

這是一個針對 NVIDIA Jetson Orin 平台設計的 CPU 效能優化教學專案。透過一系列的實驗，探討 ARM 架構下的單核/多核運算、SIMD 指令集 (Eigen) 以及記憶體頻寬對效能的影響。

## 環境需求 (Prerequisites)

在開始之前，請確保您的 Jetson 已經安裝以下開發套件：

```bash
sudo apt-get update
sudo apt-get install -y build-essential libeigen3-dev libopenblas-dev

實驗內容 (Labs)

Lab 1: 基礎向量加法 (Baseline)
檔案: lab1_vector_add.cpp
目標: 建立效能基準。使用 C++ 標準 std::vector 進行 $10^8$ 大小的向量加法，觀察單核心滿載時的行為。
編譯: g++ lab1_vector_add.cpp -o lab1 -O3

Lab 2: Eigen 最佳化與 SIMD
檔案: lab2_eigen_1e8.cpp
目標: 比較手寫迴圈與 Eigen 函式庫 (利用 ARM NEON SIMD 指令集) 的效能差異。
編譯: g++ lab2_eigen_1e8.cpp -o lab2 -O3 -I/usr/include/eigen3 -fopenmp

Lab 3: OpenBLAS 多核調度
檔案: lab3_openblas_1e8.cpp
目標: 透過調整 openblas_set_num_threads，觀察多核心在 Memory Bound (記憶體受限) 運算下的非線性效能提升。
編譯: g++ lab3_openblas_1e8.cpp -o lab3 -O3 -lopenblas

Lab 4: 矩陣運算與能效甜蜜點
檔案: lab4_matrix_eff.cpp
目標: 執行 2048x2048 矩陣乘法 (Compute Bound)，計算不同核心數下的能效比 (Efficiency)，找出 Jetson CPU 的運作甜蜜點。
編譯: g++ lab4_matrix_eff.cpp -o lab4 -O3 -lopenblas -I/usr/include/eigen3


執行與監控
sudo tegrastats --interval 500

建議在執行各個 Lab 時，開啟 tegrastats 或 Jetson Power GUI 來觀察以下指標：
1. CPU Load: 觀察單核 vs 多核的負載變化。
2. VDD_IN / VDD_CPU: 觀察功耗變化以計算能效比。

# 執行範例
./lab1
./lab2
./lab3
./lab4