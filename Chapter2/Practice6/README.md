# Normalization 的記憶體層級差異

### 題目說明
1. 利用 `cudaMallocManaged` 建立一個隨機浮點數矩陣 A (512 x 768)
2. 實作 GPU Kernel 計算 Mean 與 Std，並計算 (A−Mean)/Std
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 將矩陣逐步放大（512 → 2048 → 4096 → 8192），觀察效能是否呈線性增長
* 在 GPU Kernel Summary 中觀察 normalize_kernel 的 Duration，分析矩陣大小對 Kernel 執行時間的影響
