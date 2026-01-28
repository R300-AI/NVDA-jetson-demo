# Normalization 的記憶體層級差異

### 題目說明
1. 利用 `cudaMallocManaged` 建立一個隨機浮點數矩陣 A (512 x 768)
2. 實作 GPU Kernel 計算 Mean 與 Std，並計算 (A−Mean)/Std
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 將矩陣逐步放大（512 → 2048 → 4096 → 8192），觀察效能是否呈線性增長
* 使用 `nsys profile` 觀察 **Memory Throughput**，分析記憶體頻寬是否成為瓶頸
* 觀察 **Kernel Launch Overhead**，評估多次 Kernel 啟動的延遲影響
