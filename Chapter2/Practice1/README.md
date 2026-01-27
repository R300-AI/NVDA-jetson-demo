# CPU 與 GPU Warp的效能差異

### 題目說明
1. 利用`wp::array` 初始化兩個形狀為 [1,10^8] 的A, B向量。
2. 使用`wp::add` 取得A+B 執行結果。
3. 利用`std::chrono` 記錄整體執行時間

### 作業練習
* 計算 加速倍率 (Speedup Ratio)
 = CPU_time / GPU_time
* 減少向量長度，觀察加速倍率的變化
