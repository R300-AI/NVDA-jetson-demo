# Practice 4 – NVIDIA Jetson CPU & GPU Optimization Labs

本 Practice 針對 **NVIDIA Jetson（以 Orin 系列為主）** 平台，設計一系列 **CPU 與 GPU 效能最佳化實驗（Labs）**，透過實際程式碼與量測工具，理解在嵌入式邊緣 AI 裝置上：

* ARM CPU 的運算瓶頸與最佳化手法
* SIMD / Eigen / OpenBLAS 對效能的影響
* CUDA 核心（Kernel）設計與 GPU 平行化概念
* CPU vs GPU 在不同工作負載下的效能差異

---

## 📁 目錄結構

```text
practice4/
├─ CPU_Labs/
│  ├─ lab1_vector_add.cpp
│  ├─ lab2_eigen_1e8.cpp
│  ├─ lab3_openblas_1e8.cpp
│  ├─ lab4_matrix_eff.cpp
│  └─ README.md
│
└─ GPU_Labs/
   ├─ lab1_cpu_vs_gpu.cu
   ├─ lab2_divergence.cu
   ├─ lab3_cublas.cu
   ├─ lab4_attention.cu
   ├─ lab5_reshape.cu
   ├─ lab6_norm.cu
   ├─ lab7_neuron.cu
   ├─ lab8_activation.cu
   ├─ CMakeLists.txt
   └─ README.md
```

---

## 🧪 CPU_Labs 說明

**CPU_Labs** 聚焦於 Jetson 上的 **ARM CPU 效能分析與最佳化**，包含：

* Baseline C++ 向量 / 矩陣運算
* Eigen（利用 SIMD / 向量化）
* OpenBLAS（多執行緒與記憶體頻寬影響）
* Compute-bound vs Memory-bound 行為分析

📌 詳細實驗目的、編譯方式與執行說明請參考：
➡️ `CPU_Labs/README.md`

---

## 🚀 GPU_Labs 說明

**GPU_Labs** 透過 CUDA 程式實作，探討：

* CPU 與 GPU 計算模型差異
* Thread divergence 對效能的影響
* cuBLAS 與手寫 CUDA kernel 的差異
* Neural Network 常見算子（Activation / Norm / Attention）的基礎實作

📌 編譯方式、CMake 設定與實驗說明請參考：
➡️ `GPU_Labs/README.md`

---

## 🛠 環境需求（建議）

* NVIDIA Jetson Orin 系列（Nano / NX / AGX）
* JetPack 5.x 以上
* CUDA Toolkit（隨 JetPack 安裝）
* g++ / cmake / Eigen / OpenBLAS

---

## 📊 效能觀察建議

實驗過程中，建議搭配以下工具進行觀察：

```bash
sudo tegrastats --interval 500
```

觀察重點包含：

* CPU 使用率與核心分佈
* GPU 使用率
* 記憶體頻寬與功耗變化

---

## 🎯 學習目標總結

完成本 Practice 後，應能：

* 理解 Jetson 平台上 CPU 與 GPU 的角色分工
* 判斷工作負載適合放在 CPU 或 GPU
* 實際體會「最佳化」對效能的影響幅度
* 為後續 Edge AI / CUDA / 系統效能分析打下基礎

---

## 📌 備註

本 Practice 為教學與學習用途，程式碼以 **可讀性與實驗觀察** 為優先，非極限最佳化版本。
