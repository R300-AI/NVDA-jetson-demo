# 利用cuBLAS進行高效的線性代數運算

### 題目說明
1. 建立 2048×2048 的大型矩陣 A
2. 調用 `cublasSgemm` 計算矩陣內積 A·A
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習

* 比較手寫矩陣乘法與 cuBLAS 的效能差異
* 計算 TFLOPS = (2 × N × N × N) / (執行時間 × 10^12)
* 通過 nsys 追蹤 cuda 效能：
  * 在 CUDA API Summary 中找到 cublasSgemm 的執行時間
  * 在 Timeline View 觀察其對應的 GPU Kernel 執行區間
