#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Kernel: Bias Add + ReLU
// 這是典型的 Memory Bound 操作
__global__ void bias_relu_kernel(float* C, const float* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;

    if (idx < total_elements) {
        int row = idx / N; 
        float val = C[idx];
        float bias = b[row];
        
        // 運算邏輯很簡單，資料讀寫才是瓶頸
        val += bias;            
        C[idx] = fmaxf(0.0f, val); 
    }
}

int main() {
    int M = 2048, N = 2048;
    std::cout << "[Lab 8] 初始化記憶體..." << std::endl;

    float *d_C, *d_b;
    cudaMallocManaged(&d_C, M * N * sizeof(float));
    cudaMallocManaged(&d_b, M * sizeof(float));

    // 模擬 Lab 7 的輸出結果
    for(int i=0; i<M*N; i++) d_C[i] = (i % 2 == 0) ? 1.0f : -1.0f; // 有正有負，測試 ReLU
    for(int i=0; i<M; i++) d_b[i] = 0.5f;

    std::cout << "準備執行 Activation Function (Bias+ReLU)..." << std::endl;
    std::cout << ">> 請觀察 Tegrastats，功耗應比 Lab 7 低" << std::endl;
    std::cout << ">> 按下 Enter 開始...";
    std::cin.get();

    std::cout << "正在執行 5000 次 ReLU..." << std::endl;

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    // 因為運算太快，我們跑 5000 次讓您有時間看數據
    for(int i=0; i<5000; i++) {
        bias_relu_kernel<<<blocks, threads>>>(d_C, d_b, M, N);
    }

    cudaDeviceSynchronize();
    std::cout << "Activation 運算完成。" << std::endl;

    cudaFree(d_C); cudaFree(d_b);
    return 0;
}