# Jetson Orin 程式實作專案 (2)

這個學習資源旨在讓你體驗 **CUDA 平行運算** 在 Jetson Orin 上的開發流程。你將學會如何撰寫 `CUDA kernel`、配置 `threads/blocks`，並觀察 GPU 的效能表現，進一步理解 **CUDA 加速原理**。 

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
### matrix_mul.cu
```cpp
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matMul(const float* A,const float* B,float* C,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<N){
        float sum=0;
        for(int k=0;k<N;k++) sum+=A[row*N+k]*B[k*N+col];
        C[row*N+col]=sum;
    }
}

int main(){
    int N=256; size_t size=N*N*sizeof(float);
    float *h_A=new float[N*N],*h_B=new float[N*N],*h_C=new float[N*N];
    for(int i=0;i<N*N;i++){ h_A[i]=1.0f; h_B[i]=2.0f; }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size); cudaMalloc(&d_B,size); cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+15)/16,(N+15)/16);
    matMul<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cout<<"C[0]="<<h_C[0]<<endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}
```
### cublas_gemm.cu
```cpp
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
using namespace std;

int main(){
    int N=256; size_t size=N*N*sizeof(float);
    float *A,*B,*C;
    cudaMallocManaged(&A,size); cudaMallocManaged(&B,size); cudaMallocManaged(&C,size);
    for(int i=0;i<N*N;i++){ A[i]=1.0f; B[i]=2.0f; }

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha=1.0f,beta=0.0f;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,B,N,A,N,&beta,C,N);
    cudaDeviceSynchronize();

    cout<<"C[0]="<<C[0]<<endl;
    cublasDestroy(handle);
    cudaFree(A); cudaFree(B); cudaFree(C);
}
```
### cudnn_conv.cu
```cpp
#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
using namespace std;

int main(){
    cudnnHandle_t handle; cudnnCreate(&handle);
    cudnnTensorDescriptor_t xDesc,yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(xDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,1,32,32);
    cudnnSetFilter4dDescriptor(wDesc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,1,1,3,3);
    cudnnSetConvolution2dDescriptor(convDesc,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

    int n,c,h,w;
    cudnnGetConvolution2dForwardOutputDim(convDesc,xDesc,wDesc,&n,&c,&h,&w);
    cout<<"Output shape: "<<n<<"x"<<c<<"x"<<h<<"x"<<w<<endl;

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroy(handle);
}
```
