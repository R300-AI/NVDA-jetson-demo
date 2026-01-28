# NVIDIA GPU 演進與 CUDA 加速原理

這個學習資源旨在讓你掌握 **CUDA 程式設計** 的基本概念與 **GPU 加速** 的核心原理。你將學會如何撰寫 CUDA Kernel、使用 cuBLAS 進行高效矩陣運算，並透過 `tegrastats` 觀察 GPU 的執行狀態。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，CUDA Toolkit 已隨 JetPack 預裝。

1. 確認 CUDA 編譯器版本
```bash
nvcc --version
```

2. 確認 cuBLAS 函式庫路徑
```bash
find /usr/local/cuda -name "libcublas.so*"
```

## 編譯與執行

1. 基本編譯指令
```bash
nvcc <source_file>.cu -o <output_binary> -O3 -arch=sm_87
```
* `<source_file>.cu`：你的 CUDA 程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱
* `-O3`：開啟最高等級優化
* `-arch=sm_87`：指定 GPU 架構（見下方說明）

### 關於 `-arch` 參數

`-arch` 參數指定 GPU 架構，用於啟用硬體特定的優化指令。常見設備對應：

| 編譯參數 | 對應設備 |
|---------|---------|
| `-arch=sm_89` | RTX 4090 / 4080 |
| `-arch=sm_87` | **Jetson Orin 系列** |
| `-arch=sm_86` | RTX 3090 / 3080 / A100 |
| `-arch=sm_72` | Jetson Xavier 系列 |
| `-arch=sm_62` | Jetson TX2 |
| `-arch=sm_53` | Jetson Nano |


> **重要**：若不指定 `-arch`，編譯器使用預設值 (sm_52)，可能損失 30-50% 效能。

2. 若程式使用 cuBLAS，需額外連結函式庫
```bash
nvcc <source_file>.cu -o <output_binary> -O3 -arch=sm_87 -lcublas
```

3. 執行程式
```bash
./<output_binary>
```

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

## 硬體效能監測

執行實驗時，可開啟新的 Terminal 執行以下指令觀察 GPU 狀態：

* **GPU 使用率**
  ```bash
  tegrastats --interval 100 | grep -o 'GR3D_FREQ [0-9]\+%'
  ```
  > `GR3D_FREQ xx%` 代表 GPU 引擎使用率。

* **GPU 功耗**
  ```bash
  tegrastats --interval 100 | grep -o 'VDD_[A-Z0-9_]\+ [0-9]\+mW/[0-9]\+mW'
  ```
  > `VDD_GPU_SOC` 或 `VDD_CPU_GPU_CV` 為 GPU 相關功耗（依機型而異）。

* **記憶體使用量**
  ```bash
  tegrastats --interval 100 | grep -o 'RAM [0-9/]\+MB'
  ```
  > 觀察 GPU 運算時的記憶體變化。
