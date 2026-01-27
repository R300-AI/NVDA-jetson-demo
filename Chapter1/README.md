# Jetson Orin硬體架構與運算特性 - 快速上手C++

這個學習資源旨在讓你快速掌握 **Jetson Orin** 平台的開發流程。你將學會如何設定C++環境、編譯並執行以 `Eigen` 和 `OpenBLAS` 為基礎的矩陣運算程式，並透過 `nsight-systems` 工具即時觀察 CPU、GPU 及記憶體的使用狀況。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，如果你的系統尚未安裝 **Ubuntu with JetPack**，請先依照 [官方教學](https://developer.nvidia.com/embedded/jetpack-sdk-62)完成安裝。

1. 安裝開發工具與數學運算函式庫
  ```bash
  sudo apt update
  sudo apt install -y build-essential libeigen3-dev libopenblas-dev

  sudo apt-get update
  sudo apt-get install python3-pip python3-setuptools python3-wheel -y
  pip3 install --upgrade pip setuptools wheel
  sudo apt-get install python3-dev libffi-dev libssl-dev -y
  pip3 install jetson-stats

  git clone https://github.com/rbonghi/jetson_stats.git
  cd jetson_stats
  sudo python3 setup.py install
  jtop

  sudo apt update
  sudo apt install python3-pip
  sudo -H pip3 install -U jetson-stats
  sudo reboot
  ```
2. 確認函式庫的安裝路徑，方便後續編譯程式碼
  ```bash
  # 查找 Eigen 標頭檔路徑（通常為 /usr/include/eigen3）
  find /usr/include -type d -name "eigen3"

  # 查找 OpenBLAS 函式庫路徑（通常為 /usr/lib/aarch64-linux-gnu）
  find /usr/lib -name "libopenblas.so*"

  # 確認Nsight-Systems是否已經正確安裝
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
3. 開啟效能監測（四個題目通用，一行即可）：
```bash
sudo nsys profile -o report --force-overwrite=true --trace=osrt --sample=cpu --cpuctxsw=true ./<output_binary>
```

參數含意：
- `profile`：開始錄製效能分析，產生報告檔（`.nsys-rep`）。
- `-o report`：設定輸出檔名前綴，會輸出成 `report.nsys-rep`。
- `--force-overwrite=true`：若同名報告已存在，直接覆蓋。
- `--trace=osrt`：追蹤 OS runtime 事件（thread 建立/同步/排程等），用來觀察多執行緒行為。
- `--sample=cpu`：啟用 CPU 取樣（sampling），用來看 CPU 時間主要花在哪些函式/呼叫路徑。
- `--cpuctxsw=true`：記錄 CPU context switch（執行緒切換），用來觀察排程與切換開銷。
- `./<output_binary>`：你要量測的程式。

（可選）輸出文字摘要：
```bash
nsys stats report.nsys-rep --report summary,osrt_summary
```

（可選）如果你需要額外觀察 VDD_CPU / CPU 頻率，可同時開啟：
```bash
sudo tegrastats --interval 500
```
