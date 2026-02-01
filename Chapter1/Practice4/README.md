# Eigen 與 OpenBLAS 效能對比

### 題目說明
1. 使用 `Eigen::MatrixXf` 建立大小為 [1024 × 1024] 以上的隨機矩陣 A、B
2. 使用 Eigen 的矩陣運算符 C = A * B 完成矩陣乘法，並記錄執行時間
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
比較 Eigen 矩陣乘法與 Practice 3 (OpenBLAS) 的效能差異
