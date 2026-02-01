# 透過異質運算實踐Artificial Neurons

### 題目說明
1. 使用 cuBLAS 執行一個 2048×2048 的矩陣乘法 (GEMM)，得到結果矩陣 C
2. 建立一個長度為 2048 的偏置向量 b
3. 實作 Bias Addition Kernel，將偏置向量加到矩陣 C 的每一列：C' = C + b

### 作業練習
* 透過 `cuda` trace 產生報告，在 **CUDA API Summary** 中比較 `cublasSgemm` 與 `bias_add_kernel` 的執行時間佔比
* 在 **Timeline View** 中觀察 GEMM Kernel 與 Bias Kernel 的執行區間，確認 GEMM 為 Compute Bound 操作（執行區間較長）
* 記錄此練習的 GEMM Kernel 執行時間，供 Practice 8 比較 Compute Bound 與 Memory Bound 的差異

