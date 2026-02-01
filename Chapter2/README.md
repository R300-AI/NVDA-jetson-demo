# NVIDIA GPU 演進與 CUDA 加速原理

這個學習資源旨在讓你掌握 **CUDA 程式設計** 的基本概念與 **GPU 加速** 的核心原理。你將學會如何撰寫 CUDA Kernel、使用 cuBLAS 進行高效矩陣運算，並透過 `Nsight Systems` 進行深入的效能分析。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，Nsight Systems 需要額外在**Jetson Orin**及**工作站主機**安裝CLI及視覺化工具。

1. 確認 CUDA 編譯器與 cuBLAS 函式庫
```bash
nvcc --version
ldconfig -p | grep cublas
nsys profile --trace=cuda --capture-range=cudaProfilerApi -o lab1_test ./lab1_cpu_vs_gpu
```

### (可選) Nsight Systems 效能監測工具

2. 於**Jetson Orin**下載並安裝 [**Linux on ARM**](https://developer.nvidia.com/nsight-systems/get-started)版本的 CLI Profiler.

```bash
sudo apt install ./nsight-systems-<version>-arm64.deb
nsys --version
```
 
3. 於**Workstation**下載並安裝 [**Windows on x86_64**](https://developer.nvidia.com/nsight-systems/get-started)版本的 GUI視覺化分析工具.

    ![](https://github.com/R300-AI/NVDA-jetson-demo/blob/main/assets/nsys_example.png)

4. 通過Type-C LOG Port或Wi-Fi將Jetson Orin與主機建立連線

## 編譯與執行

1. 執行編譯指令

    ```bash
    nvcc <source_file>.cu -o <output_binary> -O2 -arch=sm_87 -lcublas
    ```
    * `<source_file>.cu`：你的 CUDA 程式碼檔案
    * `<output_binary>`：編譯後的執行檔名稱
    * `-O2`：開啟編譯優化
    * `-arch=sm_87`：將 GPU 架構指定為Jetson Orin
    * `-lcublas`：連結 cuBLAS 函式庫（Practice 3, 4, 7 需要）

    > 如果程式碼使用 Eigen，請加上 `-I /usr/include/eigen3`

2. 執行程式

    ```bash
    ./<output_binary>

    # 如果你需要額外監測硬體效能，請改用以下命令
    nsys profile --trace=cuda --capture-range=cudaProfilerApi -o <trace_name> ./<output_binary>
    ```

    | Trace 選項 | 說明 |
    |------------|------|
    | `cuda` | 記錄 CUDA API 呼叫與 Kernel 執行時間 |
    | `nvtx` | 記錄 NVTX 標記（需在程式碼中加入） |
    | `osrt` | 記錄作業系統執行緒活動 |

3. 將Profile的紀錄檔`.nsys-rep`傳到**Workstation**，並通過Windows版**Nsight Systems**開啟該檔案以觀察硬體效能

    ```
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

### CUDA Profiler API

在 Jetson 平台使用 `nsys profile` 時，若程式使用 `cudaMallocManaged` 配置大量記憶體，可能會遇到 `NvMapMemAllocInternalTagged` 錯誤。解決方法是使用 **CUDA Profiler API** 來控制 profiling 範圍，僅追蹤 Kernel 執行階段：

```cpp
#include <cuda_profiler_api.h>

// 記憶體配置與初始化（不追蹤）
cudaMallocManaged(&data, size);
init_data(data);

// 開始 Profiling（僅追蹤 GPU 計算階段）
cudaProfilerStart();
my_kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();
cudaProfilerStop();

// 釋放記憶體（不追蹤）
cudaFree(data);
```

搭配 `--capture-range=cudaProfilerApi` 參數執行：

```bash
nsys profile --trace=cuda --capture-range=cudaProfilerApi -o trace_output ./my_program
```

> ⚠️ **重要注意事項**：
> 1. `cudaProfilerStart()` 與 `cudaProfilerStop()` 之間**必須有 GPU 活動**（Kernel 執行或 CUDA API 呼叫），否則 nsys 會因無法建立 GPU 時間戳對應而發生 `TimeConversion.cpp` 錯誤
> 2. 純 CPU 運算**不應**包含在 profiling 範圍內
> 3. `cudaProfilerStart()` 應放在 GPU Kernel 呼叫**之前**，而非記憶體配置之後
>
> **錯誤示範**：
> ```cpp
> cudaProfilerStart();
> cpu_function();      // ❌ 純 CPU 運算，GPU 閒置
> gpu_kernel<<<...>>>(); 
> cudaProfilerStop();
> ```
>
> **正確示範**：
> ```cpp
> cpu_function();      // CPU 運算放在外面
> cudaProfilerStart();
> gpu_kernel<<<...>>>(); // ✅ 立即有 GPU 活動
> cudaDeviceSynchronize();
> cudaProfilerStop();
> ```

> **注意**：本章所有練習的 `.cu` 檔案已內建 Profiler API，可直接使用上述指令進行效能分析。