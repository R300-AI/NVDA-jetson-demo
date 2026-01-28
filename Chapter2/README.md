# NVIDIA GPU 演進與 CUDA 加速原理

這個學習資源旨在讓你掌握 **CUDA 程式設計** 的基本概念與 **GPU 加速** 的核心原理。你將學會如何撰寫 CUDA Kernel、使用 cuBLAS 進行高效矩陣運算，並透過 `Nsight Systems` 進行深入的效能分析。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，CUDA Toolkit 已隨 JetPack 預裝。

1. 確認 CUDA 編譯器與 cuBLAS 函式庫
```bash
nvcc --version
find /usr/local/cuda -name "libcublas.so*"
```

2. 安裝 NVIDIA Nsight Systems
```bash
# 請從 https://developer.nvidia.com/nsight-systems/get-started 下載對應版本
sudo apt install ./nsight-systems-<version>-arm64.deb
nsys --version
```

## 編譯與執行

1. 執行編譯指令
```bash
nvcc <source_file>.cu -o <output_binary> -O3 -arch=sm_87 -lcublas
```
* `<source_file>.cu`：你的 CUDA 程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱
* `-O3`：開啟最高等級優化
* `-arch=sm_87`：指定 GPU 架構

    | 編譯參數 | 對應設備 |
    |---------|---------|
    | `-arch=sm_89` | RTX 4090 / 4080 |
    | `-arch=sm_87` | **Jetson Orin 系列** |
    | `-arch=sm_86` | RTX 3090 / 3080 / A100 |
    | `-arch=sm_72` | Jetson Xavier 系列 |
    | `-arch=sm_62` | Jetson TX2 |
    | `-arch=sm_53` | Jetson Nano |

2. 執行程式
```bash
./<output_binary>
```

3. 如果你需要額外觀察硬體效能，可使用以下工具：
    ```bash
    # 記錄程式的執行過程
    nsys profile -o report ./your_program

    # 通過GUI 查看分析報告
    nsys-ui report.nsys-rep
    ```

    | 指標 | 說明 |
    |-----|------|
    | **CUDA Kernel 時間軸** | 每個 Kernel 的啟動時間與執行時長，可識別效能瓶頸 |
    | **GPU Utilization** | GPU 運算單元的實際使用率 |
    | **Warp Stall Reasons** | Warp 停滯原因（如 Divergence、Memory Wait） |
    | **Memory Throughput** | 記憶體讀寫頻寬，識別 Memory Bound 問題 |
    | **CUDA API Calls** | cuBLAS、cudaMalloc 等 API 呼叫時間 |
    | **Kernel Launch Overhead** | Kernel 啟動延遲，評估是否需要合併 Kernel |
    | **Memory Operations** | Host-Device 資料傳輸，確認 Zero-Copy 是否生效 |
    | **SM Occupancy** | 每個 SM 的執行緒佔用率，評估平行化效率 |



## CUDA 程式設計基礎

### Kernel 函數與執行配置

CUDA Kernel 是在 GPU 上執行的函數，使用 `__global__` 修飾符宣告：

```cpp
__global__ void my_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

啟動 Kernel 時需指定執行緒配置：
```cpp
int threads = 256;                          // 每個 Block 的執行緒數
int blocks = (N + threads - 1) / threads;   // Block 數量
my_kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();                    // 等待 GPU 完成
```

### Managed Memory（統一記憶體）

Jetson 平台上 CPU 與 GPU 共用實體記憶體，使用 Managed Memory 可簡化資料管理：

```cpp
float *data;
cudaMallocManaged(&data, N * sizeof(float));  // 配置統一記憶體
// ... CPU 和 GPU 皆可存取 data ...
cudaFree(data);                               // 釋放記憶體
```

### cuBLAS 矩陣乘法

cuBLAS 是 NVIDIA 官方優化的線性代數函式庫，`cublasSgemm` 用於單精度矩陣乘法：

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
// C = alpha * A * B + beta * C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,              // 矩陣維度
            &alpha,
            d_A, lda,             // 矩陣 A 與 leading dimension
            d_B, ldb,             // 矩陣 B 與 leading dimension
            &beta,
            d_C, ldc);            // 矩陣 C 與 leading dimension

cublasDestroy(handle);
```

> **注意**：cuBLAS 使用 **Column-major** 儲存格式，與 C/C++ 預設的 Row-major 不同。