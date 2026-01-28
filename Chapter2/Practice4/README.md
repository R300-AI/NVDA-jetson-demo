# 透過GPU加速Self Attention

### 題目說明
1. 建立一個 d=768、標記長度 N=512 的 Token 矩陣
2. 使用 `cublasSgemm` 及 `CUBLAS_OP_T`，計算 Q×K^T，輸出 [N,N] 中間矩陣 S
3. 將矩陣 S 除以 √d，並利用提供的 `softmax_scaling_kernel` 計算 P
4. 使用 `cublasSgemm` 計算 P×V，得到 Self-Attention 的輸出

### 作業練習
* 將標記長度 N 從 512 提升至 2048，計算中間矩陣 [N,N] 的大小 (MB)
* 使用 `nsys profile` 觀察 **CUDA Kernel 時間軸**，分析 GEMM 與 Softmax 各佔多少時間比例
* 觀察 **CUDA API Calls** 中各 `cublasSgemm` 的執行時間，確認哪個步驟是瓶頸
