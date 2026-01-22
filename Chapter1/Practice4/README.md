# 最佳化流程設計

### 題目說明
1. 使用 `Eigen::MatrixXf 及setRandom()`建立大小為 [2048 × 2048] 的隨機矩陣 A、B，
2. 呼叫 `openblas_set_num_threads(n)` 調整執行緒數量（例如 n = 1, 2, 4, 8）
3. 利用 `cblas_sgemm()` 完成乘法運算C = A × B
4. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
比較不同執行緒數量下的Efficiency，找出 CPU 運作效率最高的「甜蜜點」

Efficiency = ⁡1/ (𝐸𝑥𝑒𝑐𝑢𝑡𝑖𝑜𝑛 𝑇𝑖𝑚𝑒×𝑉𝐷𝐷_𝐶𝑃𝑈)
