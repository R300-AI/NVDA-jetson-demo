#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// 簡單的 Softmax Kernel (用於 Lab 4)
__global__ void softmax_scaling_kernel(float* S, int N, float scale) {
    int row = blockIdx.x;
    if (row < N) {
        float max_val = -1e9;
        // 1. Find Max
        for (int i=0; i<N; i++) max_val = fmaxf(max_val, S[row*N + i]);
        
        // 2. Exp sum
        float sum = 0.0f;
        for (int i=0; i<N; i++) {
            S[row*N + i] = expf((S[row*N + i] - max_val) * scale);
            sum += S[row*N + i];
        }
        
        // 3. Normalize
        for (int i=0; i<N; i++) S[row*N + i] /= sum;
    }
}

int main() {
    int N = 2048; // Token Length
    int d = 768;  // Dimension
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Q, K, V, S, Output 記憶體配置 (使用 Managed Memory)
    float *d_Q, *d_K, *d_V, *d_S, *d_Out;
    cudaMallocManaged(&d_Q, N * d * sizeof(float));
    cudaMallocManaged(&d_K, N * d * sizeof(float)); // 假設 K 已經轉置或由 cuBLAS 處理
    cudaMallocManaged(&d_V, N * d * sizeof(float));
    cudaMallocManaged(&d_S, N * N * sizeof(float)); // 中間矩陣 [N, N]
    cudaMallocManaged(&d_Out, N * d * sizeof(float));

    // 初始化略...

    float alpha = 1.0f, beta = 0.0f;
    float scale = 1.0f / sqrtf((float)d);

    auto start = std::chrono::high_resolution_clock::now();

    // 1. S = Q * K^T (使用 cuBLAS)
    // 注意: cuBLAS 是 Column-major，這裡假設我們處理好了數據佈局
    // Q: [N, d], K: [N, d] -> S: [N, N]
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, d, 
                &alpha, d_K, d, d_Q, d, 
                &beta, d_S, N);

    // 2. Softmax & Scaling (Custom Kernel)
    softmax_scaling_kernel<<<N, 1>>>(d_S, N, scale);

    // 3. Out = S * V
    // S: [N, N], V: [N, d] -> Out: [N, d]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                d, N, N, 
                &alpha, d_V, d, d_S, N, 
                &beta, d_Out, d);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Attention Layer Time: " << (end-start).count()/1e9 << " s" << std::endl;
    std::cout << "Intermediate Matrix Size: " << (N * N * 4) / (1024.0*1024.0) << " MB" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_Out);
    return 0;
}