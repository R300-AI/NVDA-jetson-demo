# 透過異質運算實踐Artificial Neurons

### 題目說明
1. 使用 cuBLAS 執行一個 2048×2048 的矩陣乘法 (GEMM)，得到結果矩陣 C
2. 建立一個長度為 2048 的偏置向量 b
3. 實作 Bias Addition Kernel，將偏置向量加到矩陣 C 的每一列：C' = C + b

### 作業練習
* 使用 `nsys profile` 觀察 **CUDA API Calls**，比較 GEMM 與 Bias Addition 的時間佔比
* 觀察 **CUDA Kernel 時間軸**，確認 GEMM 為 Compute Bound 操作（執行時間長、GPU 利用率高）
* 記錄此練習的總執行時間，供 Practice 8 比較
Artificial Neurons

### 題目說明
1. 使用 cuBLAS 執行一個 2048×2048 的矩陣乘法 (GEMM)，得到結果矩陣 C。
2. 建立一個長度為 2048 的偏置向量，並將偏置向量 b 加到矩陣 C 的每一列：
C′=C+b
### 作業練習
* 使用 tegrastats 觀察 POM_5V_GPU 欄位，記錄 GPU 在執行 GEMM 時的功耗。
* 同時使用 htop 或tegrastats查看 CPU及GPU在異質運算下的處理情況。

