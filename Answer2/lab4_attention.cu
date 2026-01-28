#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Softmax Kernel: 對 S 矩陣的每一列進行 Softmax
__global__ void softmax_scaling_kernel(float* S, int N, float scale) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    // 1. 找最大值 (數值穩定性)
    float max_val = -1e9f;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, S[row * N + i] * scale);
    }
    
    // 2. 計算 exp 並求和
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        S[row * N + i] = expf(S[row * N + i] * scale - max_val);
        sum += S[row * N + i];
    }
    
    // 3. 正規化
    for (int i = 0; i < N; i++) {
        S[row * N + i] /= sum;
    }
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 GEMM vs Softmax 時間軸" << std::endl;

    const int N = 512;   // Token 長度
    const int d = 768;   // Embedding 維度
    
    std::cout << "Token 長度 N = " << N << ", 維度 d = " << d << std::endl;

    // 配置記憶體: Q, K, V: [N, d], S: [N, N], Out: [N, d]
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

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: S = Q × K^T
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, d,
                &alpha,
                d_K, d,
                d_Q, d,
                &beta,
                d_S, N);
    cudaDeviceSynchronize();

    // Step 2: P = Softmax(S / √d)
    softmax_scaling_kernel<<<N, 1>>>(d_S, N, scale);
    cudaDeviceSynchronize();

    // Step 3: Out = P × V
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, N, N,
                &alpha,
                d_V, d,
                d_S, N,
                &beta,
                d_Out, d);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // ========== 輸出結果 ==========
    std::cout << "Attention Layer Time: " << diff.count() << " s" << std::endl;
    
    // 中間矩陣大小
    double s_size_mb = (N * N * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "中間矩陣 S 大小: " << s_size_mb << " MB" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_Out);
    return 0;
}
