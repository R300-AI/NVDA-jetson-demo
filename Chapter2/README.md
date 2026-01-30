# NVIDIA GPU 演進與 CUDA 加速原理

這個學習資源旨在讓你掌握 **CUDA 程式設計** 的基本概念與 **GPU 加速** 的核心原理。你將學會如何撰寫 CUDA Kernel、使用 cuBLAS 進行高效矩陣運算，並透過 `Nsight Systems` 進行深入的效能分析。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，Nsight Systems 需要額外在**Jetson Orin**及**工作站主機**安裝CLI及視覺化工具。

1. 確認 CUDA 編譯器與 cuBLAS 函式庫
```bash
nvcc --version
find /usr/local/cuda -name "libcublas.so*"
```

### (可選) Nsight Systems 效能監測工具 - 2024.5.1相容版

2. 於**Jetson Orin**下載並安裝 [**Linux on ARM**](https://developer.nvidia.com/nsight-systems/get-started)版本的 CLI Profiler.

```bash
sudo apt install ./nsight-systems-<version>-arm64.deb
nsys --version
```
 
2. 於**Workstation**下載並安裝 [**Windows on x86_64**](https://developer.nvidia.com/nsight-systems/get-started)版本的 GUI視覺化分析工具.

    ![](https://github.com/R300-AI/NVDA-jetson-demo/blob/main/assets/nsys_example.png)

3. 通過Type-C LOG Port或Wi-Fi將Jetson Orin與主機建立連線

## 編譯與執行

1. 執行編譯指令

```bash
nvcc <source_file>.cu -o <output_binary> -O2 -arch=sm_87 -lcublas
```
* `<source_file>.cu`：你的 CUDA 程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱
* `-O2`：開啟編譯優化
* `-arch=sm_87`：指定 GPU 為Jetson Orin架構

2. 執行程式

```bash
./<output_binary>

#如果你需要額外監測硬體效能，請改用以下命令
nsys profile --trace=cuda -o <trace_name> ./<output_binary>
```
    |Trace 選項| 說明|
    |-----|------|
    | `cuda` | xx |
    | `cuda` | xx |

3. 將Profile的紀錄檔`.nsys-rep`傳到**Workstation**，並通過Windows版**Nsight Systems**開啟該檔案以觀察硬體效能

```
scp <jetson_orin_user>@<jetson_orin_ip>:<path_to_nsys-rep_file> <host_path>
```


## CUDA 程式設計基礎

本節提供完成各 Practice 所需的各種核心技巧。

### 技巧 1：記憶體配置與釋放

Jetson 平台上 CPU 與 GPU 共用實體記憶體，使用 Managed Memory 可簡化資料管理：

```cpp
// 配置 Managed Memory（CPU/GPU 皆可存取）
float *data;
size_t bytes = N * sizeof(float);
cudaMallocManaged(&data, bytes);

// 使用完畢後釋放
cudaFree(data);
```

---

### 技巧 2：Kernel 函數與執行配置

CUDA Kernel 使用 `__global__` 修飾符，透過 `<<<blocks, threads>>>` 語法啟動：

```cpp
// 定義 Kernel（在 GPU 上執行）
__global__ void my_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 計算全域索引
    if (idx < N) {                                     // 邊界檢查
        data[idx] = data[idx] * 2.0f;
    }
}

// 啟動 Kernel
int threads = 256;                          // 每個 Block 的執行緒數（建議 128~512）
int blocks = (N + threads - 1) / threads;   // Block 數量（向上取整）
my_kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();                    // 等待 GPU 完成
```

> **適用練習**：P1, P2, P6, P7, P8

---

### 技巧 3：指標與 Eigen 整合（Zero-Copy）

在 Managed Memory 上使用 `Eigen::Map` 可避免資料複製：

```cpp
#include <Eigen/Dense>

// 配置 Managed Memory
float* ptr;
cudaMallocManaged(&ptr, rows * cols * sizeof(float));

// 使用 Eigen::Map 建立矩陣視圖（不複製資料）
Eigen::Map<Eigen::MatrixXf> mat(ptr, rows, cols);
mat.setRandom();  // 可用 Eigen 的方法操作

// Reshape 為向量（同樣不複製資料）
Eigen::Map<Eigen::VectorXf> vec(ptr, rows * cols);

// 驗證 Zero-Copy：位址應相同
std::cout << "矩陣位址: " << ptr << std::endl;
std::cout << "向量位址: " << &vec(0) << std::endl;  // 應與上行相同
```

> **適用練習**：P5, P6

---

### 技巧 4：cuBLAS 矩陣乘法

cuBLAS 是 NVIDIA 官方優化的線性代數函式庫：

```cpp
#include <cublas_v2.h>

// 建立 Handle
cublasHandle_t handle;
cublasCreate(&handle);

// 矩陣乘法 C = α×A×B + β×C
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,  // 是否轉置（N=不轉置, T=轉置）
            M, N, K,                    // 維度：C[M,N] = A[M,K] × B[K,N]
            &alpha,
            d_A, M,                     // 矩陣 A 與 leading dimension
            d_B, K,                     // 矩陣 B 與 leading dimension
            &beta,
            d_C, M);                    // 矩陣 C 與 leading dimension

// 釋放 Handle
cublasDestroy(handle);
```

> **注意**：cuBLAS 使用 **Column-major** 格式，與 C/C++ 預設的 Row-major 不同。  
> **適用練習**：P3, P4, P7

---

### 技巧 5：條件分支與 Warp Divergence

同一 Warp 內的 32 個執行緒應盡量走相同分支：

```cpp
// ❌ 不良寫法：每個 Warp 內有一半走不同分支
__global__ void bad_kernel(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % 2 == 0) {
        C[tid] = A[tid] + B[tid];  // 偶數
    } else {
        C[tid] = A[tid] - B[tid];  // 奇數
    }
}

// ✓ 改良寫法：重新排列任務，讓同一 Warp 走相同分支
__global__ void good_kernel(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;
    if (tid < half) {
        C[tid * 2] = A[tid * 2] + B[tid * 2];      // 前半段處理偶數
    } else {
        int idx = (tid - half) * 2 + 1;
        C[idx] = A[idx] - B[idx];                   // 後半段處理奇數
    }
}
```

> **適用練習**：P2

---

### 技巧 6：逐列操作（Normalization / Softmax）

對矩陣的每一列進行獨立運算時，可讓每個 Block 處理一列：

```cpp
// 每個 Block 處理一列
__global__ void normalize_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;  // 每個 Block 負責一列
    if (row >= rows) return;
    
    float* row_ptr = data + row * cols;  // 該列的起始指標
    
    // Step 1: 計算平均值
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum += row_ptr[i];
    }
    float mean = sum / cols;
    
    // Step 2: 計算標準差
    float sq_sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float diff = row_ptr[i] - mean;
        sq_sum += diff * diff;
    }
    float std_dev = sqrtf(sq_sum / cols + 1e-5f);
    
    // Step 3: 正規化
    for (int i = 0; i < cols; i++) {
        row_ptr[i] = (row_ptr[i] - mean) / std_dev;
    }
}

// 啟動時 blocks = rows
normalize_kernel<<<rows, 1>>>(data, rows, cols);
```

> **適用練習**：P4 (Softmax), P6 (Normalization)

---

### 技巧 7：向量廣播（Bias Addition）

將偏置向量加到矩陣的每一列：

```cpp
// C'[i][j] = C[i][j] + b[i]
__global__ void bias_add_kernel(float* C, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;  // 計算該元素屬於哪一列
        C[idx] = C[idx] + b[row];
    }
}

// 啟動配置
int threads = 256;
int blocks = (rows * cols + threads - 1) / threads;
bias_add_kernel<<<blocks, threads>>>(C, b, rows, cols);
```

> **適用練習**：P7

---

### 技巧 8：元素級操作（Activation Functions）

ReLU 等 Activation 是典型的 Memory Bound 操作：

```cpp
// ReLU(x) = max(0, x)
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```