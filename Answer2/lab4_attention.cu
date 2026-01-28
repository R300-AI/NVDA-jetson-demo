/*
 * Practice 4: 透過 GPU 加速 Self Attention
 * 
 * 題目要求：
 * 1. 建立一個 d=768、標記長度 N=512 的 Token 矩陣
 * 2. 使用 cublasSgemm 及 CUBLAS_OP_T 計算 Q×K^T，輸出 [N,N] 中間矩陣 S
 * 3. 將矩陣 S 除以 √d，並利用提供的 softmax_scaling_kernel 計算 P
 * 4. 使用 cublasSgemm 計算 P×V，得到 Self-Attention 的輸出
 * 
 * 編譯: nvcc lab4_attention.cu -o lab4 -O2 -arch=sm_87 -lcublas
 * 執行: ./lab4
 * 監測: nsys profile -o report_lab4 ./lab4
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Softmax Kernel: 對 S 矩陣的每一列進行 Softmax
// 每個 Block 處理一列
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
    const int N = 512;   // Token 長度
    const int d = 768;   // Embedding 維度
    
    std::cout << "=== Practice 4: Self Attention ===" << std::endl;
    std::cout << "Token 長度 N = " << N << ", 維度 d = " << d << std::endl;

    // 配置記憶體
    // Q, K, V: [N, d]
    // S (中間矩陣): [N, N]
    // Out: [N, d]
    float *Q, *K, *V, *S, *Out;
    cudaMallocManaged(&Q, N * d * sizeof(float));
    cudaMallocManaged(&K, N * d * sizeof(float));
    cudaMallocManaged(&V, N * d * sizeof(float));
    cudaMallocManaged(&S, N * N * sizeof(float));
    cudaMallocManaged(&Out, N * d * sizeof(float));

    // 初始化 Q, K, V (模擬 embedding)
    for (int i = 0; i < N * d; i++) {
        Q[i] = 0.01f;
        K[i] = 0.01f;
        V[i] = 0.01f;
    }

    // 建立 cuBLAS Handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    float scale = 1.0f / sqrtf((float)d);  // 1/√d

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: S = Q × K^T
    // Q: [N, d] (row-major) -> cuBLAS 視為 [d, N] (col-major)
    // K: [N, d] (row-major) -> cuBLAS 視為 [d, N] (col-major)
    // S: [N, N]
    // 
    // 在 cuBLAS 中: S = K^T × Q (因為 column-major)
    // CUBLAS_OP_T 對 K 做轉置
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,  // K 轉置, Q 不轉置
                N, N, d,                    // M=N, N=N, K=d
                &alpha,
                K, d,                       // K: lda = d
                Q, d,                       // Q: ldb = d  
                &beta,
                S, N);                      // S: ldc = N
    cudaDeviceSynchronize();

    // Step 2: P = Softmax(S / √d)
    softmax_scaling_kernel<<<N, 1>>>(S, N, scale);
    cudaDeviceSynchronize();

    // Step 3: Out = P × V
    // P: [N, N], V: [N, d] -> Out: [N, d]
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, N, N,                    // M=d, N=N, K=N
                &alpha,
                V, d,                       // V: lda = d
                S, N,                       // P (stored in S): ldb = N
                &beta,
                Out, d);                    // Out: ldc = d
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // ========== 輸出結果 ==========
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\n--- 結果 ---" << std::endl;
    std::cout << "Attention 執行時間: " << elapsed << " s" << std::endl;
    
    // 中間矩陣大小
    double s_size_mb = (N * N * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "中間矩陣 S 大小: " << s_size_mb << " MB" << std::endl;
    
    // 如果 N=2048，S 大小會是多少?
    int N2 = 2048;
    double s2_size_mb = (N2 * N2 * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "若 N=2048，S 大小: " << s2_size_mb << " MB" << std::endl;

    cublasDestroy(handle);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(S);
    cudaFree(Out);
    return 0;
}
