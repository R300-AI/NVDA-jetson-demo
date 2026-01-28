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

// ========== 高度分歧 Kernel (效能較差) ==========
// 偶數 thread 做運算，奇數 thread 不做 -> 導致整個 Warp 等待
__global__ void divergent_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx % 2 == 0) { 
            data[idx] = heavy_math(data[idx]);
        } else {
            data[idx] = data[idx]; // 奇數位置不做運算
        }
    }
}

// ========== TODO: 實作無分歧 Kernel (效能較佳) ==========
// 重新排列任務，讓執行運算的 thread 聚在一起
// 提示：前半段 thread 做事 -> 整個 Warp 要嘛全做，要嘛全不做
__global__ void optimized_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* 請實作無分歧版本的 Kernel */
    
}

int main() {
    std::cout << "【實驗提示】" << std::endl;
    std::cout << "請提前開啟 tegrastats，觀察:" << std::endl;
    std::cout << "1. GR3D_FREQ (GPU 使用率)" << std::endl;
    std::cout << "2. VDD_GPU 功耗數值" << std::endl;

    // ========== TODO 1: 設定向量大小 ==========
    
    int N = 1000;                   /* 請填入正確的向量大小 (建議 10^8) */
    
    
    // ========== 配置記憶體 ==========
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));
    
    // 初始化
    for(int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;


    // ========== 測試 Divergent Kernel ==========
    auto start_div = std::chrono::high_resolution_clock::now();
    
    divergent_kernel<<<blocks, threads>>>(data, N);
    cudaDeviceSynchronize();
    
    auto end_div = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_div = end_div - start_div;
    

    // ========== TODO 2: 測試 Optimized Kernel ==========
    // 請仿照上方的計時方式，測試 optimized_kernel 的執行時間

    /* 請填入 Optimized Kernel 測試程式碼 */
    std::chrono::duration<double> t_opt = std::chrono::duration<double>(0);  // 暫時設為 0


    // ========== 輸出結果 ==========
    std::cout << "Divergent Time: " << t_div.count() << " s" << std::endl;
    std::cout << "Optimized Time: " << t_opt.count() << " s" << std::endl;
    

    // ========== TODO 3: 計算效能損失 ==========
    // 效能損失 = (T_divergence - T_optimized) / T_divergence * 100%

    /* 請計算並輸出效能損失百分比 */


    // ========== 釋放記憶體 ==========
    cudaFree(data);
    
    return 0;
}
