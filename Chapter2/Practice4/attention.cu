#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>

// ========== Softmax Scaling Kernel ==========
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
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "使用 nsys profile 監測，觀察 GEMM vs Softmax 時間軸" << std::endl;

    // ========== TODO 1: 建立一個 d=768、標記長度 N=512 的 Token 矩陣 ==========

    int N = 128;                    /* 請填入正確的 Token 長度 (512) */
    int d = 64;                     /* 請填入正確的 Embedding 維度 (768) */
    
    std::cout << "Token 長度 N = " << N << ", 維度 d = " << d << std::endl;

    // 配置記憶體: Q, K, V: [N, d], S: [N, N], Out: [N, d]
    float *d_Q, *d_K, *d_V, *d_S, *d_Out;
    cudaMallocManaged(&d_Q, N * d * sizeof(float));
    /* 請接著配置 d_K, d_V, d_S, d_Out 的記憶體 */


    // ========== 初始化 Q, K, V (隨機值模擬) ==========
    for (int i = 0; i < N * d; i++) {
        d_Q[i] = 0.1f;
        d_K[i] = 0.1f;
        d_V[i] = 0.1f;
    }


    // ========== 建立 cuBLAS Handle ==========
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    float scale = 1.0f / sqrtf((float)d);


    // ========== 開始計時 ==========
    auto start = std::chrono::high_resolution_clock::now();


    // ========== TODO 3: 計算 S = Q × K^T ==========
    // 使用 cublasSgemm，注意 K 需要轉置 (CUBLAS_OP_T)
    // Q: [N, d], K: [N, d] -> S: [N, N]
    
    /* 請填入 Q × K^T 的 cublasSgemm 程式碼 */


    // ========== TODO 4: Softmax & Scaling ==========
    // 使用提供的 softmax_scaling_kernel

    /* 請填入 Softmax Kernel 執行程式碼 */


    // ========== TODO 5: 計算 Out = P × V ==========
    // S: [N, N], V: [N, d] -> Out: [N, d]

    /* 請填入 P × V 的 cublasSgemm 程式碼 */


    cudaDeviceSynchronize();


    // ========== 結束計時 ==========
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    // ========== 輸出結果 ==========
    std::cout << "Attention Layer Time: " << diff.count() << " s" << std::endl;
    

    // ========== TODO 6: 計算中間矩陣大小 ==========
    // 中間矩陣 S 的大小 = N * N * sizeof(float) / (1024 * 1024) MB

    /* 請計算並輸出中間矩陣大小 (MB) */


    // ========== 清理資源 ==========
    cublasDestroy(handle);
    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_S); 
    cudaFree(d_Out);
    
    return 0;
}
