# GPU Warp Divergence 之效能分析

### 題目說明
1. 利用 `wp::array` 初始化兩個形狀為 [1, 10^8] 的 A, B 向量
2. 利用 `wp::launch` 在 GPU 上執行下列運算：
   * 若 tid 為偶數，執行 C[tid] = A[tid] + B[tid]
   * 若 tid 為奇數，執行 C[tid] = A[tid] - B[tid]
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 計算效能損失 = (T_divergence − T_optimized) / T_divergence
* 使用 `nsys profile` 觀察報告中的 **Warp Stall Reasons**，確認 Divergence 造成的停滯
* 比較 Divergent 與 Optimized Kernel 在 **SM Occupancy** 的差異
