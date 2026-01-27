# Jetson Orin硬體架構與運算特性 - 快速上手C++

這個學習資源旨在讓你快速掌握 **Jetson Orin** 平台的開發流程。你將學會如何設定C++環境、編譯並執行以 `Eigen` 和 `OpenBLAS` 為基礎的矩陣運算程式，並透過 `tegrastats` 工具即時觀察 CPU、GPU 及記憶體的使用狀況。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，如果你的系統尚未安裝 **Ubuntu with JetPack**，請先依照 [官方教學](https://developer.nvidia.com/embedded/jetpack-sdk-62)完成安裝。

1. 安裝開發工具與數學運算函式庫
  ```bash
  sudo apt update
  sudo apt install -y build-essential libeigen3-dev libopenblas-dev
  ```
2. 確認函式庫的安裝路徑，方便後續編譯程式碼
  ```bash
  # 查找 Eigen 標頭檔路徑（通常為 /usr/include/eigen3）
  find /usr/include -type d -name "eigen3"

  # 查找 OpenBLAS 函式庫路徑（通常為 /usr/lib/aarch64-linux-gnu）
  find /usr/lib -name "libopenblas.so*"
  ```

## 編譯與執行

1. 使用你查詢到的 Eigen 標頭檔路徑與 OpenBLAS 函式庫路徑，執行編譯指令
```bash
g++ <source_file>.cpp -o <output_binary> -I <eigen_include_path> -L <openblas_lib_path> -lopenblas
```
* `<source_file>.cpp`：你的程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱
* `<eigen_include_path>`：Eigen 標頭檔路徑
* `<openblas_lib_path>`：OpenBLAS 函式庫路徑

2. 執行程式：
```bash
./<output_binary>
```

3. 如果你需要額外觀察 VDD_CPU / CPU 頻率，可同時開啟：
```bash
tegrastats --interval 100
01-27-2026 11:15:02 RAM 2250/7620MB (lfb 3x4MB) SWAP 0/3810MB (cached 0MB) CPU [66%@1728,16%@1728,18%@1728,9%@1728,9%@729,16%@729] GR3D_FREQ 0% cpu@45.968C soc2@45.156C soc0@45.468C gpu@45.437C tj@45.968C soc1@44.281C VDD_IN 4046mW/4236mW VDD_CPU_GPU_CV 720mW/786mW VDD_SOC 1201mW/1261mW


# CPU 使用率 (每個核心)
tegrastats --interval 100 | grep -o 'CPU \[[^]]*\]'

CPU [0%@729,16%@729,0%@729,0%@729,0%@729,0%@729]

# GPU 使用率 (GR3D_FREQ)
tegrastats --interval 100 | grep -o 'GR3D_FREQ [0-9]\+%'

GR3D_FREQ 0%

# RAM 使用量
tegrastats --interval 100 | grep -o 'RAM [0-9/]\+MB'

RAM 2258/7620MB (lfb 3x4MB

# SWAP 使用量
tegrastats --interval 100 | grep -o 'SWAP [0-9/]\+MB'

SWAP 0/3810MB (cached 0MB

# 溫度 (AO@, GPU@, CPU@ 等)
tegrastats --interval 100 | grep -o '[a-z0-9]\+@[0-9.]\+C'

cpu@45.406C
soc2@44.718C
soc0@45.156C
gpu@44.906C
tj@45.406C
soc1@43.781C


# 功耗 (POM_5V_IN, POM_5V_GPU, POM_5V_CPU 等)
tegrastats --interval 100 | grep -o 'VDD_[A-Z0-9_]\+ [0-9]\+mW/[0-9]\+mW'

VDD_IN 4086mW/4086mW
VDD_CPU_GPU_CV 720mW/707mW
VDD_SOC 1241mW/1241mW

```
