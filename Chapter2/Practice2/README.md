# GPU Warp Divergence 之效能分析

## 題目說明
1. 利用`wp::array`初始化兩個形狀為 [1,10^8] 的A, B向量。
2. 利用 `wp::launch`在 GPU 上執行下列運算
  * 若tid 為偶數，執行C[tid] = A[tid] + B[tid]
  * 若tid 為奇數，執行C[tid] = A[tid]  - B[tid]
3. 利用 `std::chrono` 記錄整體執行時間

## 作業練習
* 計算效能損失 = 實作2之1 − Tdivergence) / Tdivergence
* 透過 tegrastats的GR3D_FREQ 與 VDD_GPU 觀察GPU利用率是否有差異或下降趨勢
