# NVIDIA GPU 演進與 CUDA 加速原理

這個學習資源旨在讓你掌握 **CUDA 程式設計** 的基本概念與 **GPU 加速** 的核心原理。你將學會如何撰寫 CUDA Kernel、使用 cuBLAS 進行高效矩陣運算，並透過 `Nsight Systems` 進行深入的效能分析。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，Nsight Systems 需要額外在**Jetson Orin**及**工作站主機**安裝CLI及視覺化工具。

1. 確認 CUDA 編譯器與 cuBLAS 函式庫
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    ```bash
    nvcc --version
    ldconfig -p | grep cublas
    ```

3. 安裝 Nsight Systems（Jetson Orin）

    ```bash
    # 搜尋可用版本
    apt-cache search nsight-systems

    # 安裝 JetPack 6.2 相容版本（通常是 2024.5.x）
    sudo apt update
    sudo apt install nsight-systems-2024.5

    # 確認版本
    nsys --version
    ```

4. 安裝 Nsight Systems GUI（工作站）

於 **Workstation** 下載並安裝 [**Windows on x86_64**](https://developer.nvidia.com/nsight-systems/get-started) 版本的 GUI 視覺化分析工具。

![](https://github.com/R300-AI/NVDA-jetson-demo/blob/main/assets/nsys_example.png)

4. 建立連線

通過 Type-C LOG Port 或 Wi-Fi 將 Jetson Orin 與主機建立連線。

## 編譯與執行

1. 執行編譯指令

```bash
nvcc <source_file>.cu -o <output_binary> -O2 -arch=sm_87 -lcublas
```

| 參數 | 說明 |
|------|------|
| `<source_file>.cu` | 你的 CUDA 程式碼檔案 |
| `<output_binary>` | 編譯後的執行檔名稱 |
| `-O2` | 開啟編譯優化 |
| `-arch=sm_87` | 將 GPU 架構指定為 Jetson Orin |
| `-lcublas` | 連結 cuBLAS 函式庫（Practice 3, 4, 7 需要）|

2. 執行程式

```bash
# 直接執行
./<output_binary>

# 使用 Nsight Systems 進行效能監測，產生 Profile 的紀錄檔
nsys profile --trace=cuda -o <trace_name> ./<output_binary>
```

3. 傳輸與分析

將 Profile 的紀錄檔 `.nsys-rep` 傳到 **Workstation**，並通過 Windows 版 **Nsight Systems** 開啟該檔案以觀察硬體效能：

```bash
scp <jetson_orin_user>@<jetson_orin_ip>:<path_to_nsys-rep_file> <host_path>
```

## CUDA 程式設計基礎

本節提供完成各 Practice 所需的核心概念。

### 記憶體配置（Managed Memory）

Jetson 平台上 CPU 與 GPU 共用實體記憶體，使用 Managed Memory 可簡化資料管理：

```cpp
float *data;
size_t bytes = N * sizeof(float);
cudaMallocManaged(&data, bytes);  // CPU/GPU 皆可存取
cudaFree(data);                   // 使用完畢後釋放
```

### Kernel 執行配置

CUDA Kernel 使用 `__global__` 修飾符，透過 `<<<blocks, threads>>>` 語法啟動：

```cpp
int threads = 256;                          // 每個 Block 的執行緒數（建議 128~512）
int blocks = (N + threads - 1) / threads;   // Block 數量（向上取整）
my_kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();                    // 等待 GPU 完成
```

### cuBLAS 基本用法

cuBLAS 是 NVIDIA 官方優化的線性代數函式庫：

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);
// ... 執行運算 ...
cublasDestroy(handle);
```
