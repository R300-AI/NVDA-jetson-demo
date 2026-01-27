# 透過OpenBLAS實踐多核矩陣乘法

### 題目說明
1. 建立大小為 [1024 × 1024] 的隨機矩陣 A、B（使用 `std::vector<float>` 及 `std::generate`）
2. 呼叫 `openblas_set_num_threads(n)` 調整執行緒數量（例如 n = 1, 2, 4, 8）
3. 使用 `cblas_sgemm()` 完成矩陣乘法運算 C = A × B
4. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
觀察並比較不同執行緒數量下的執行時間變化，分析多核資源調度的非線性效能提升：
- 觀察 `CPU [xx%@freq, ...]` 的各核心使用率/頻率變化
- 監測 `VDD_<rail> current/avg`（例如 `VDD_CPU_GPU_CV`）功耗輸出情況
- 比較單核與多核的加速比（Speedup = 單核時間 / 多核時間）
- 理解矩陣乘法的計算密度高，更適合展現多核平行優勢
