# CPU 與 GPU 的效能差異

### 題目說明
1. 利用 `cudaMallocManaged` 初始化兩個形狀為 [1, 10^7] 的 A, B 向量
2. 分別使用 CPU for-loop 與 GPU Kernel 計算 A + B
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 計算加速倍率 (Speedup Ratio) = CPU_time / GPU_time
* 逐步增加向量長度（10^6 → 10^7 → 5×10^7），觀察加速倍率的變化
* 使用 `nsys profile` 觀察報告中的 **CUDA Kernel 時間軸** 與 **GPU Utilization**，比較 CPU 與 GPU 的執行時間佔比
