#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// 模擬重負載運算
__device__ float heavy_math(float x) {
    for(int i = 0; i < 100; i++) {
        x = x * x + 0.001f;
    }
    return x;
}

// 高度分歧 Kernel (效能較差)
// 偶數 thread 做運算，奇數 thread 不做 -> 導致整個 Warp 等待
__global__ void divergent_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx % 2 == 0) { 
            data[idx] = heavy_math(data[idx]);
        } else {
            data[idx] = data[idx];
        }
    }
}

// 無分歧 Kernel (效能較佳)
// 重新排列任務，讓執行運算的 thread 聚在一起
// 前半段 thread 做事 -> 整個 Warp 要嘛全做，要嘛全不做
__global__ void optimized_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        data[idx] = heavy_math(data[idx]);
    }
}

int main() {
    int N = 100000000; // 10^8
    
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));
    
    // 初始化
    for(int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    divergent_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();

    // ========== 測試 Divergent Kernel ==========
    auto start_div = std::chrono::high_resolution_clock::now();
    divergent_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();
    auto end_div = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_div = end_div - start_div;

    // 重新初始化
    for(int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    // Warmup
    optimized_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();

    // ========== 測試 Optimized Kernel ==========
    auto start_opt = std::chrono::high_resolution_clock::now();
    optimized_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_opt = end_opt - start_opt;

    // ========== 輸出結果 ==========
    std::cout << "Divergent Time: " << t_div.count() << " s" << std::endl;
    std::cout << "Optimized Time: " << t_opt.count() << " s" << std::endl;
    
    // 計算效能損失
    double loss = (t_div.count() - t_opt.count()) / t_div.count() * 100.0;
    std::cout << "效能損失 (Divergence 造成): " << loss << "%" << std::endl;

    cudaFree(data);
    return 0;
}
