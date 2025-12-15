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
