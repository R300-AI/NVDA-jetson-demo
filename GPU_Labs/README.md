# NVIDIA Jetson Orin - GPU Optimization & CUDA Programming Labs

這是一個針對 NVIDIA Jetson Orin 平台 (Architecture: Ampere/sm_87) 設計的進階 GPU 程式設計與效能優化教學專案。涵蓋了從基礎的 CUDA Kernel 撰寫、記憶體管理、Warp Divergence 分析，到結合 cuBLAS 與 Eigen 進行深度學習運算加速的實作。

## 環境需求 (Prerequisites)

請確保 Jetson 開發板已安裝 JetPack，並執行以下指令安裝必要的 C++ 數學庫：

```bash
sudo apt-get update
sudo apt-get install -y libeigen3-dev

CUDA Toolkit: 預設已隨 JetPack 安裝。
Eigen3: 用於矩陣初始化與 CPU 端運算對照。

實驗列表與編譯指令
為了獲得最佳效能，所有編譯指令皆開啟 -O3 優化，並針對 Orin 架構設定 -arch=sm_87。

基礎運算與架構分析
Lab 1: CPU vs GPU 向量加法
檔案: lab1_cpu_vs_gpu.cu
說明: 比較 CPU 單核與 CUDA 多執行緒在向量加法上的效能差異，使用 Managed Memory 簡化記憶體管理。
編譯: nvcc lab1_cpu_vs_gpu.cu -o lab1 -O3 -arch=sm_87
觀察: 比較 CPU Time 與 GPU Time 的加速倍率。

Lab 2: Warp Divergence 效能分析
檔案: lab2_divergence.cu
說明: 展示分支發散 (Warp Divergence) 對 GPU 效能的影響。比較「分歧核心 (Divergent)」與「合併核心 (Coalesced/Optimized)」的執行時間。
編譯: nvcc lab2_divergence.cu -o lab2 -O3 -arch=sm_87

進階矩陣運算 (cuBLAS)
Lab 3: 高效矩陣乘法 (cuBLAS GEMM)
檔案: lab3_cublas.cu
說明: 使用 NVIDIA 官方高度優化的 cuBLAS 函式庫進行 SGEMM (矩陣乘法) 運算，計算 TFLOPS。
編譯 (需連結 cuBLAS 庫): nvcc lab3_cublas.cu -o lab3 -O3 -arch=sm_87 -lcublas

Lab 4: Self-Attention 機制實作
檔案: lab4_attention.cu
說明: 模擬 Transformer 模型中的 Self-Attention 層 (Q x K^T x V)，結合 cuBLAS 與自定義 Softmax Kernel。
編譯: nvcc lab4_attention.cu -o lab4 -O3 -arch=sm_87 -lcublas

記憶體優化與 Zero-Copy
Lab 5: 透過指標減少 Reshape 搬移
檔案: lab5_reshape.cu
說明: 利用 Unified Memory 與 Eigen Map 實現 Zero-Copy Reshape，證明改變矩陣形狀無需搬移實體記憶體。
編譯 (需抑制 Eigen 相關警告): nvcc lab5_reshape.cu -o lab5 -O3 -arch=sm_87 -I/usr/include/eigen3 -diag-suppress 20011,20012

Lab 6: Normalization 記憶體層級差異
檔案: lab6_norm.cu
說明: 實作 Layer Normalization 的簡化版本，觀察 GPU 在處理統計量計算時的記憶體存取模式。
編譯: nvcc lab6_norm.cu -o lab6 -O3 -arch=sm_87 -I/usr/include/eigen3

異質運算與功耗分析
Lab 7: Artificial Neurons (Compute Bound)
檔案: lab7_neuron.cu
說明: 執行高強度的矩陣乘法，模擬神經元權重計算。此為「運算密集型」任務。
編譯: nvcc lab7_neuron.cu -o lab7 -O3 -arch=sm_87 -lcublas
觀察: 執行時請開啟 tegrastats，觀察 VDD_IN 功耗應顯著上升。

Lab 8: Activation Functions (Memory Bound)
檔案: lab8_activation.cu
說明: 執行 Bias Addition 與 ReLU 激活函數。此為「記憶體密集型」任務，瓶頸在於 VRAM 頻寬。
編譯: nvcc lab8_activation.cu -o lab8 -O3 -arch=sm_87
觀察: 比較此實驗與 Lab 7 的功耗差異 (通常此實驗功耗較低)。

效能監控工具
建議在執行實驗時，另開終端機使用 tegrastats 或 Jetson Power GUI 觀察即時硬體狀態： # 每 500ms 更新一次數據
sudo tegrastats --interval 500

觀察重點欄位：
GR3D_FREQ: GPU 使用率與頻率。
VDD_IN / VDD_GPU_CV: 整體與 GPU 功耗 (mW)。
RAM: 系統記憶體佔用量。