# Chapter 2 練習題解答

本資料夾包含 Chapter 2 所有練習題的正確解答，對應各 Practice 的題目要求。

## 編譯指令

所有程式碼使用以下指令編譯（使用 `-O2` 優化）：

```bash
# Lab 1, 2, 5, 6, 8 (純 CUDA)
nvcc labX_xxx.cu -o labX -O2 -arch=sm_87

# Lab 3, 4, 7 (需要 cuBLAS)
nvcc labX_xxx.cu -o labX -O2 -arch=sm_87 -lcublas
```

## 使用 Nsight Systems 進行效能分析

```bash
# 記錄程式執行過程
nsys profile -o report_labX ./labX

# 查看分析報告
nsys-ui report_labX.nsys-rep
```

## 各練習對應檔案

| 練習 | 檔案 | 編譯指令 | 觀察重點 |
|------|------|----------|----------|
| P1 | `lab1_cpu_vs_gpu.cu` | `nvcc lab1_cpu_vs_gpu.cu -o lab1 -O2 -arch=sm_87` | CUDA Kernel 時間軸、GPU Utilization |
| P2 | `lab2_divergence.cu` | `nvcc lab2_divergence.cu -o lab2 -O2 -arch=sm_87` | Warp Stall Reasons、SM Occupancy |
| P3 | `lab3_cublas.cu` | `nvcc lab3_cublas.cu -o lab3 -O2 -arch=sm_87 -lcublas` | CUDA API Calls |
| P4 | `lab4_attention.cu` | `nvcc lab4_attention.cu -o lab4 -O2 -arch=sm_87 -lcublas` | Kernel 時間軸 (GEMM vs Softmax) |
| P5 | `lab5_reshape.cu` | `nvcc lab5_reshape.cu -o lab5 -O2 -arch=sm_87` | Memory Operations (Zero-Copy) |
| P6 | `lab6_norm.cu` | `nvcc lab6_norm.cu -o lab6 -O2 -arch=sm_87` | Memory Throughput、Kernel Launch Overhead |
| P7 | `lab7_neuron.cu` | `nvcc lab7_neuron.cu -o lab7 -O2 -arch=sm_87 -lcublas` | CUDA API Calls (GEMM vs Bias) |
| P8 | `lab8_activation.cu` | `nvcc lab8_activation.cu -o lab8 -O2 -arch=sm_87` | Memory Throughput、GPU Utilization |

## 執行範例

```bash
# 編譯 Lab 1
nvcc lab1_cpu_vs_gpu.cu -o lab1 -O2 -arch=sm_87

# 直接執行
./lab1

# 使用 Nsight Systems 記錄
nsys profile -o report_lab1 ./lab1

# 查看報告
nsys-ui report_lab1.nsys-rep
```
