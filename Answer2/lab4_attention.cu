#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>

// Softmax Scaling Kernel
// 對每一列進行 Softmax 並乘上 scale 係數
__global__ void softmax_scaling_kernel(float* S, int N, float scale) {
    int row = blockIdx.x;
    if (row < N) {
        float max_val = -1e9;
        
        // 1. 找最大值 (數值穩定性)
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, S[row * N + i]);
        }
        
        // 2. 計算 exp 並求和
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            S[row * N + i] = expf((S[row * N + i] - max_val) * scale);
            sum += S[row * N + i];
        }
        
        // 3. 正規化
        for (int i = 0; i < N; i++) {
            S[row * N + i] /= sum;
        }
    }
}

int main() {
    int N = 512;   // Token 長度
    int d = 768;   // Embedding 維度
    
    std::cout << "Token 長度 N = " << N << ", 維度 d = " << d << std::endl;

    // 配置記憶體
    float *d_Q, *d_K, *d_V, *d_S, *d_Out;
    cudaMallocManaged(&d_Q, N * d * sizeof(float));
    cudaMallocManaged(&d_K, N * d * sizeof(float));
    cudaMallocManaged(&d_V, N * d * sizeof(float));
    cudaMallocManaged(&d_S, N * N * sizeof(float));
    cudaMallocManaged(&d_Out, N * d * sizeof(float));

    // 初始化 Q, K, V
    for (int i = 0; i < N * d; i++) {
        d_Q[i] = 0.1f;
        d_K[i] = 0.1f;
        d_V[i] = 0.1f;
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    float scale = 1.0f / sqrtf((float)d);

    // 計時開始
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: S = Q × K^T
    // Q: [N, d], K: [N, d] -> S: [N, N]
    // cuBLAS 是 Column-major，所以實際計算 S = K^T × Q
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, d, 
                &alpha, d_K, d, d_Q, d, 
                &beta, d_S, N);

    // Step 2: Softmax & Scaling
    softmax_scaling_kernel<<<N, 1>>>(d_S, N, scale);

    // Step 3: Out = S × V (實際是 P × V)
    // S: [N, N], V: [N, d] -> Out: [N, d]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                d, N, N, 
                &alpha, d_V, d, d_S, N, 
                &beta, d_Out, d);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Attention Layer Time: " << diff.count() << " s" << std::endl;
    
    // 計算中間矩陣大小
    double intermediate_size_mb = (N * N * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "中間矩陣 S 大小: " << intermediate_size_mb << " MB" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_S); 
    cudaFree(d_Out);
    return 0;
}
