# Eigen 矩陣乘法與 OpenBLAS 單核對比

### 題目說明
1. 使用 `Eigen::MatrixXf` 建立大小為 [1024 × 1024] 的隨機矩陣 A、B
2. 使用 Eigen 的矩陣運算符 `C = A * B` 完成矩陣乘法，並記錄執行時間
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
比較 Eigen 矩陣乘法與 Practice 3 (OpenBLAS) 的效能差異：
- 觀察 `CPU [xx%@freq, ...]` 確認 Eigen 是否也使用單核心執行
- 比較 Eigen 與 OpenBLAS 單核 (n=1) 的執行時間
- 思考：為什麼矩陣乘法的結果可能與向量加法（Practice 1 vs 2）不同？
- 延伸：嘗試修改矩陣大小（512, 2048）觀察性能變化趨勢
