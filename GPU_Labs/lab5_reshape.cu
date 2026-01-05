#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>

// 簡單的內積 Kernel
__global__ void dot_product(const float* vec, float* result, int len) {
    // 簡化版：僅示範存取，實際內積需要 Reduction
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len) {
        atomicAdd(result, vec[idx] * vec[idx]);
    }
}

int main() {
    // Jetson 特有：使用 HostAlloc (Pinned Memory) 或 Managed 讓 CPU/GPU 指標互通
    float* ptr;
    cudaMallocManaged(&ptr, 1024 * 768 * sizeof(float));

    // 1. 使用 Eigen Map 在該指標上建立矩陣 (View)
    Eigen::Map<Eigen::MatrixXf> mat(ptr, 1024, 768);
    mat.setRandom();

    std::cout << "原始矩陣首位址: " << ptr << std::endl;

    // 2. Reshape: 在 Eigen 中改變 View，不改變底層數據
    // 注意: Eigen 預設是 Column-Major，Reshape 行為需注意
    Eigen::Map<Eigen::VectorXf> vec(ptr, 1024 * 768);
    
    std::cout << "Reshape 後首位址: " << &vec(0) << std::endl;

    if (ptr == &vec(0)) {
        std::cout << "驗證成功：位址相同，無資料搬移 (Zero-Copy)" << std::endl;
    }

    // 3. 直接丟進 Kernel 計算
    float* d_result;
    cudaMallocManaged(&d_result, sizeof(float));
    *d_result = 0;
    
    dot_product<<<2048, 512>>>(ptr, d_result, 1024*768); // 直接使用 ptr
    cudaDeviceSynchronize();

    std::cout << "Dot Product Result: " << *d_result << std::endl;

    cudaFree(ptr);
    cudaFree(d_result);
    return 0;
}