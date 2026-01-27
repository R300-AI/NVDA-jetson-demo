# 透過Eigen學習現代化程式設計

### 題目說明
1. 使用 `Eigen::VectorXf` 建立大小為 [1, 10^8] 的隨機向量 A、B
2. 直接使用 Eigen 的向量運算符 `C = A + B` 完成向量加法
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
比較 Eigen 與 Practice 1 (C++ for-loop) 的執行時間差異：
- 觀察 `CPU [xx%@freq, ...]` 確認是否仍為單核心執行
- 比較執行時間，理解 Eigen 在簡單向量加法下的效能表現
- 監測 `VDD_<rail> current/avg`（例如 `VDD_CPU_GPU_CV`）CPU 功耗輸出情況
- 體驗 Eigen 的程式碼簡潔性優勢（`C = A + B` vs for-loop）

