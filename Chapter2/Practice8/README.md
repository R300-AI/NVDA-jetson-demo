# Activation Functions的處理效能

### 題目說明
1. 承接 Practice 7 的結果矩陣 C'（或重新建立 2048×2048 矩陣模擬）
2. 實作 ReLU Kernel，在 GPU 上計算 R = max(0, C')
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 在 **CUDA GPU Kernel Summary** 中觀察 `relu_kernel` 的平均 **Duration**，確認其為 Memory Bound 操作（執行時間短但受限於記憶體頻寬）
* 在 **Timeline View** 中比較 Practice 7 (GEMM) 與 Practice 8 (ReLU) 的 Kernel 執行區間長度差異
* 觀察 1000 次 ReLU 迭代在 **Timeline View** 中的密集排列，理解 Memory Bound 操作的特性
