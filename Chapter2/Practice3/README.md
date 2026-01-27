# 利用cuBLAS進行高效的線性代數運算

### 題目說明
1. 執行講師提供的2048×2048 大型矩陣乘法範例A
2. 調用 `cublasSgemm`計算矩陣內積A·A
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 比較手寫矩陣乘法與 cuBLAS 的效能差異
* 在執行運算時，利用 tegrastats 記錄 GPU 的功耗變化（VDD_GPU）。
