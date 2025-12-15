# Jetson Orin 程式實作專案 (2)

這個學習資源旨在讓你體驗 **TensorRT** 與 DLA 加速 在 Jetson Orin 上的開發流程。你將學會如何載入 ONNX 模型、選擇 GPU 或 DLA 執行推論，並觀察 FP32、FP16 與 INT8 的效能差異，進一步理解 **DLA 架構定位**與**量化理論**。
## 準備環境 

確認系統已安裝 CUDA Toolkit（JetPack 已內建）。 
```
bash nvcc --version
 ```

## 編譯與執行

1. 使用 `nvcc` 編譯 CUDA 程式：

```bash
nvcc <source_file>.cu -o <output_binary>
```
2. 執行程式：

```bash
./<output_binary>
```
3. 啟動效能監測（Nsight Systems）：
```bash
nsys profile ./<source_file>
```

## 進階練習題

1. 請撰寫一個程式，分別使用 CPU 與 CUDA kernel 完成 100×100 矩陣的「加法」與「乘法」運算（C = A + B, C = A × B），並使用 `tegrastats` 測量 Jetson Orin 的 GPU 使用量，觀察 CPU 與 GPU 的效能差異。

2. 請使用 cuBLAS 實作 100×100 矩陣的「乘法」運算（C = A × B），並比較與自行撰寫 CUDA kernel 的執行時間，並使用 `Nsight Systems` 分析 CPU 與 GPU 的互動時間線。

3. 延續第二題，請改變矩陣大小（例如 200×200、500×500），分別比較 CUDA kernel 與 cuBLAS 的執行時間差異，並使用 `tegrastats` 觀察 GPU 的使用量，理解矩陣大小對效能的影響。

## 範例解答
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
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for(int i=0;i<N;i++){ h_A[i]=1.0f; h_B[i]=2.0f; }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size); cudaMalloc(&d_B,size); cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;
    vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cout<<"C[0]="<<h_C[0]<<endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}
```
