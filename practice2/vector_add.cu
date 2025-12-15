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

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cout<<"C[0]="<<h_C[0]<<endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}
