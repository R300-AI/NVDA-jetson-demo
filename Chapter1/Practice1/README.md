# 透過C++實作一個簡單的向量加法

### 題目說明
1. 利用 `std::vector<float>` 及 `std::generate` 建立大小為 [1, 10^8] 的隨機向量 A、B
2. 使用 for-loop 計算向量加法 C = A + B
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
觀察 CPU 單核心運作情況，作為後續比較的基準線：
- 觀察 `CPU [xx%@freq, ...]` 中單一核心的最高使用率與頻率
- 監測 `VDD_<rail> current/avg`（例如 `VDD_CPU_GPU_CV`）執行期間的 CPU 功耗輸出峰值
- 記錄執行時間，作為 Practice 2 的對比基准
