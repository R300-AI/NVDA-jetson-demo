# 透過Eigen最佳化單核的效能

### 題目說明
1. 使用 `Eigen::VectorXf` 及 `setRandom()` 建立大小為 [1, 10^8]的隨機向量𝐴, 𝐵 
2. 直接使用 C = A + B 完成向量加法運算
3. 利用 `std::chrono` 記錄整體執行時間

###作業練習
比較 Eigen 多核加法 與 C++ for-loop 單核加法 的執行時間差異，並監測 VDD_CPU 功耗輸出情況
