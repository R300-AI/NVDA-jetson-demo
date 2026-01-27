# 透過GPU加速Self Attention

### 題目說明
1. 建立一個 d=768、標記長度 N=512 的 Token 矩陣
2. 使用 cublasSgemm 及 CUBLAS_OP_T，計算 Q×KT，並輸出的 [N,N] 中間矩陣 S
3. 將矩陣 S 除以 d，並利用講師提供的 Softmax_Scaling_Kernel 計算 P。
4. 使用 cublasSgemm 計算 P×V，得到 Self-Attention 的輸出。

### 作業練習
* 將標記長度 N 從 512 提升至 2048，觀察中間矩陣 [N,N] 的大小為多少MB
* 觀察中間矩陣超過 SRAM 容量而回寫至 VRAM 時，執行時間呈線性增長?
