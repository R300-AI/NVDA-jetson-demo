# Jetson Orin 程式實作專案 (2)

這個學習資源旨在讓你體驗 **CUDA 平行運算** 在 Jetson Orin 上的開發流程。你將學會如何撰寫 `CUDA kernel`、配置 `threads/blocks`，並觀察 GPU 的效能表現，進一步理解簡報中提到的 **CUDA 加速原理**。

## 準備環境

確認系統已安裝 CUDA Toolkit（JetPack 已內建）。

```bash
nvcc --version
```
建立一個新的專案資料夾，準備撰寫 .cu 檔案。


## 編譯與執行

1. 使用 nvcc 編譯 CUDA 程式：

```bash
nvcc vector_add.cu -o vector_add
```
2. 執行程式：

```bash
./vector_add
```
3. 啟動效能監測（可於另一個命令列執行）：
```bash
sudo tegrastats
```

## 進階練習題

1. 向量加法 CUDA kernel
  * 撰寫一個 kernel 完成 C = A + B
  * 比較 CPU 與 GPU 的執行時間差異
2. 矩陣乘法 CUDA kernel
  * 撰寫一個 kernel 完成 C = A × B
  * 嘗試不同的 threads/blocks 配置，觀察效能差異
3. 延伸挑戰：混合精度運算
  * 將矩陣乘法改用 float 與 half 精度
  * 比較效能與結果精度的差異


## 範例程式：向量加法
### vector_add.cu
```cpp
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 1<<20;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = new float[N]; h_B = new float[N]; h_C = new float[N];
    for(int i=0;i<N;i++){ h_A[i]=1.0f; h_B[i]=2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cout << "C[0] = " << h_C[0] << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
```
