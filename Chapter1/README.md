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

1. 執行編譯指令
```bash
g++ <source_file>.cpp -I /usr/include/eigen3 -lopenblas -o <output_binary>
```
* `<source_file>.cpp`：你的程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱

2. 執行程式
```bash
./<output_binary>
```

3. 如果你需要額外觀察硬體效能，可開啟新的 Terminal 執行以下指令：

  * **CPU 使用率 (每個核心)**
    ```bash
    tegrastats --interval 100 | grep -o 'CPU \[[^]]*\]'
    ```
    > `CPU [xx%@freq, ...]`（`xx%`=各核心使用率；`freq`=該核心頻率 MHz）。

  * **GPU 使用率 (GR3D_FREQ)**
    ```bash
    tegrastats --interval 100 | grep -o 'GR3D_FREQ [0-9]\+%'
    ```
    > `GR3D_FREQ xx%`，代表 GPU（GR3D 引擎）使用率。

  * **RAM 使用量**
    ```bash
    tegrastats --interval 100 | grep -o 'RAM [0-9/]\+MB'
    ```
    > `RAM used/totalMB`（已用/總 RAM，單位 MB）。

  * **SWAP 使用量**
    ```bash
    tegrastats --interval 100 | grep -o 'SWAP [0-9/]\+MB'
    ```
    > `SWAP used/totalMB`（已用/總 SWAP，單位 MB；used 上升代表開始換頁）。

  * **溫度 (AO@, GPU@, CPU@ 等)**
    ```bash
    tegrastats --interval 100 | grep -o '[a-z0-9]\+@[0-9.]\+C'
    ```
    > `<sensor>@<temp>C`（°C）；`cpu@`=CPU、`gpu@`=GPU、`tj@`=晶片熱點/整體指標、`soc0/1/2@`=SoC 各區域、`ao@`=Always-On 電源域。

  * **功耗 (POM_5V_IN, POM_5V_GPU, POM_5V_CPU 等)**
    ```bash
    tegrastats --interval 100 | grep -o 'VDD_[A-Z0-9_]\+ [0-9]\+mW/[0-9]\+mW'
    ```
    > `VDD_<rail> current/avg`（mW；current=瞬時、avg=平均）；`VDD_IN`=整機輸入、`VDD_CPU_GPU_CV`=CPU+GPU+CV 域、`VDD_SOC`=SoC 其餘主要域。
