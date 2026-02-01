#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

// 填充隨機數 [0, 1)
static void fill_random_uniform_0_1(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (float)std::rand() / RAND_MAX;
    }
}

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
    std::cout << "使用 nsys profile --capture-range=cudaProfilerApi 監測" << std::endl;

    // ========== TODO 1: 建立一個 d=768、標記長度 N=512 的 Token 矩陣 ==========
    int N = 512;                    /* 請將 Token 長度改為 512 */
    int d = 768;                    /* 請將 Embedding 維度改為 768 */
    std::cout << "Token 長度 N = " << N << ", 維度 d = " << d << std::endl;

    // 配置記憶體: Q, K, V: [N, d], S: [N, N], Out: [N, d]
    float *d_Q, *d_K, *d_V, *d_S, *d_Out;
    cudaMallocManaged(&d_Q, N * d * sizeof(float));
    /* 請使用 cudaMallocManaged 配置 d_K, d_V, d_S, d_Out 的記憶體
       提示: d_K 與 d_V 大小為 N * d * sizeof(float)
             d_S 大小為 N * N * sizeof(float)
             d_Out 大小為 N * d * sizeof(float)
    */
    cudaMallocManaged(&d_K, N * d * sizeof(float));
    cudaMallocManaged(&d_V, N * d * sizeof(float));
    cudaMallocManaged(&d_S, N * N * sizeof(float));
    cudaMallocManaged(&d_Out, N * d * sizeof(float));

    // ========== 初始化 Q, K, V ==========
    std::srand(42);
    fill_random_uniform_0_1(d_Q, N * d);
    fill_random_uniform_0_1(d_K, N * d);
    fill_random_uniform_0_1(d_V, N * d);

    // ========== 建立 cuBLAS Handle ==========
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    float scale = 1.0f / sqrtf((float)d);

    // ========== 開始 Profiling（僅追蹤 GPU 計算）==========
    cudaProfilerStart();

    // ========== 計時開始 ==========
    auto start = std::chrono::high_resolution_clock::now();

    // ========== TODO 2: S = Q × K^T ==========
    /* 請呼叫 cublasSgemm 計算 S = Q × K^T
       提示: S[N,N] = Q[N,d] × K^T[d,N]
       cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d,
                   &alpha, d_K, d, d_Q, d, &beta, d_S, N)
    */
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, d,
                &alpha,
                d_K, d,
                d_Q, d,
                &beta,
                d_S, N);

    cudaDeviceSynchronize();

    // ========== TODO 3: P = Softmax(S / √d) ==========
    /* 請呼叫 softmax_scaling_kernel<<<N, 1>>>(d_S, N, scale) */
    softmax_scaling_kernel<<<N, 1>>>(d_S, N, scale);

    cudaDeviceSynchronize();

    // ========== TODO 4: Out = P × V ==========
    /* 請呼叫 cublasSgemm 計算 Out = P × V
       提示: Out[N,d] = P[N,N] × V[N,d]
       cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
                   &alpha, d_V, d, d_S, N, &beta, d_Out, d)
    */
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
    
    // ========== TODO 5: 計算中間矩陣大小 ==========
    /* 請計算並輸出中間矩陣 S[N,N] 的大小 (MB)
       提示: size_mb = (N * N * sizeof(float)) / (1024.0 * 1024.0)
    */
    double s_size_mb = (N * N * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "中間矩陣 S 大小: " << s_size_mb << " MB" << std::endl;

    // ========== 停止 Profiling ==========
    cudaProfilerStop();

    cublasDestroy(handle);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_Out);
    return 0;
}
