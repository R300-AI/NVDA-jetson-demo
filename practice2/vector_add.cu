#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // TODO: 計算 thread 的全域索引 i
    // TODO: 完成 C[i] = A[i] + B[i]
}

int main() {
    const int N = 1<<20;
    size_t size = N * sizeof(float);

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for(int i=0;i<N;i++){ h_A[i]=1.0f; h_B[i]=2.0f; }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size); cudaMalloc(&d_B,size); cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    // TODO: 設定 threadsPerBlock 與 blocksPerGrid
    // TODO: 呼叫 kernel

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cout<<"C[0]="<<h_C[0]<<endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}
