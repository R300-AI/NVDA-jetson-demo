#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// 模擬重負載運算
__device__ float heavy_math(float x) {
    for(int i=0; i<100; i++) x = x * x + 0.001f;
    return x;
}

// 1. 高度分歧 Kernel (Bad Performance)
// 偶數 thread 做事，奇數不做 -> 導致整個 Warp 等待
__global__ void divergent_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx % 2 == 0) { 
            data[idx] = heavy_math(data[idx]);
        } else {
            data[idx] = data[idx]; // Do nothing roughly
        }
    }
}

// 2. 無分歧 Kernel (Good Performance)
// 前半段 thread 做事 -> 整個 Warp 要嘛全做，要嘛全不做
__global__ void coalesced_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N/2) { // 重新排列任務，讓執行的 thread 聚在一起
        data[idx] = heavy_math(data[idx]);
    }
}

int main() {
    int N = 100000000;
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 測試 Divergent
    auto start_div = std::chrono::high_resolution_clock::now();
    divergent_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();
    auto end_div = std::chrono::high_resolution_clock::now();

    // 測試 Optimized
    auto start_opt = std::chrono::high_resolution_clock::now();
    coalesced_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();
    auto end_opt = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_div = end_div - start_div;
    std::chrono::duration<double> t_opt = end_opt - start_opt;

    std::cout << "Divergent Time: " << t_div.count() << " s" << std::endl;
    std::cout << "Optimized Time: " << t_opt.count() << " s" << std::endl;
    std::cout << "效能損失 (Loss): " << (t_div.count() - t_opt.count()) / t_div.count() * 100.0 << "%" << std::endl;

    cudaFree(data);
    return 0;
}