# Eigen 矩陣乘法與 OpenBLAS 單核對比

### 題目說明
1. 使用 `Eigen::MatrixXf` 及 `setRandom()` 建立大小為 [1024 × 1024] 的隨機矩陣 A、B
2. 使用 Eigen 的矩陣運算符 `C = A * B` 完成矩陣乘法，並記錄執行時間
3. 將相同矩陣資料轉換為一維陣列，使用 `cblas_sgemm()` 完成矩陣乘法
4. 設定 `openblas_set_num_threads(1)` 確保 OpenBLAS 使用單核心執行
5. 比較 Eigen 與 OpenBLAS 單核的執行時間差異

### 作業練習
分析 Eigen 與 OpenBLAS 單核在矩陣乘法上的性能差異：
- 觀察 `CPU [xx%@freq, ...]` 確認兩者都是單核執行
- 比較執行時間並計算速度比（Eigen時間 / OpenBLAS時間）
- 思考：為什麼矩陣乘法的結果可能與向量加法不同？
- 延伸：嘗試修改矩陣大小（512, 2048）觀察性能變化趨勢
