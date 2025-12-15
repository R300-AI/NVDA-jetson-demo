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
