# Jetson Orin 硬體架構與運算特性

這個學習資源旨在讓你快速掌握 **Jetson Orin** 平台的開發流程。你將學會如何設定環境、編譯並執行以 `Eigen` 和 `OpenBLAS` 為基礎的C++矩陣運算程式，並透過 `tegrastats` 工具即時觀察 CPU、GPU 及記憶體的使用狀況。

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
3. 開啟效能監測（可於另一個命令列執行）：
```bash
sudo tegrastats
```
