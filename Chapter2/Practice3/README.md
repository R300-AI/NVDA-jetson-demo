# 利用cuBLAS進行高效的線性代數運算

### 題目說明
1. 建立 2048×2048 的大型矩陣 A
2. 調用 `cublasSgemm` 計算矩陣內積 A·A
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 比較手寫矩陣乘法與 cuBLAS 的效能差異
* 使用 `nsys profile` 觀察報告中的 **CUDA API Calls**，分析 `cublasSgemm` 的執行時間
* 計算 TFLOPS = (2 × N × N × N) / (執行時間 × 10^12)
