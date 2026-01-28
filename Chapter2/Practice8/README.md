# Activation Functions的處理效能

### 題目說明
1. 承接 Practice 7 的結果矩陣 C'（或重新建立 2048×2048 矩陣模擬）
2. 實作 ReLU Kernel，在 GPU 上計算 R = max(0, C')
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 使用 `nsys profile` 觀察 **Memory Throughput**，確認 ReLU 為 Memory Bound 操作
* 比較 Practice 7 (GEMM) 與 Practice 8 (ReLU) 在 **Kernel 時間軸** 上的差異
* 觀察 **GPU Utilization**，對比 Compute Bound (P7) 與 Memory Bound (P8) 的 GPU 利用率差異
