# GPU Warp Divergence 之效能分析

### 題目說明
1. 利用 `cudaMallocManaged` 初始化兩個形狀為 [1, 10^7] 的 A, B 向量
2. 實作 GPU Kernel 執行下列運算：
   * 若 tid 為偶數，執行 C[tid] = A[tid] + B[tid]
   * 若 tid 為奇數，執行 C[tid] = A[tid] - B[tid]
3. 利用 `std::chrono` 記錄整體執行時間

### 作業練習
* 計算效能損失 = (T_divergence − T_optimized) / T_divergence
* 通過 `nsys` 追蹤 `cuda` 效能：
  * 在GPU Kernel Summary 中比較 divergent_kernel與 optimized_kernel的 Duration 差異
  * 在Timeline View中觀察兩個 Kernel 的執行區間長度，確認 Warp Divergence 對效能的影響
