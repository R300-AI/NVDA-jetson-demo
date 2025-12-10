# Jetson Orin 程式實作專案 - Jetpack

在這個學習資源中，使用者將學到如何使用 `gcc/g++` 編譯 `Eigen` 與 `OpenBLAS` 函式庫的程式碼，以此完成基本的矩陣與數值運算；並透過 `tegrastats` 工具觀察 CPU、GPU 與記憶體在執行過程的效能表現。

## 準備環境

在開始編譯與測試之前，先確認系統環境。以下步驟依照 JetPack 6.2 的預設狀態進行：

1. 更新套件清單
  ```bash
  sudo apt update
  ```
2. 安裝必要工具與函式庫
  ```bash
  sudo apt install -y build-essential libeigen3-dev libopenblas-dev
  ```
3. 確認編譯器版本
  ```bash
  gcc --version
  g++ --version
  ```
4. 檢查數學函式庫的安裝路徑
  ```bash
  ls /usr/include/eigen3                           # Eigen
  ls /usr/lib/aarch64-linux-gnu | grep openblas    # OpenBLAS
  ```
5. 檢查效能監測工具
```
sudo tegrastats
```
成功看到 CPU/GPU/記憶體的即時狀況後，即可按 `Ctrl+C` 結束監測。

## 執行範例

編譯指令（JetPack 6.2 預設路徑）：
```bash
g++ <source_file>.cpp -o <output_binary> -I<eigen_include_path> -L<openblas_lib_path> -lopenblas
```
* <source_file>.cpp：你的程式碼檔案。
* <output_binary>：編譯後的執行檔名稱。
* -I<eigen_include_path>：Eigen 的標頭檔路徑，例如 /usr/include/eigen3。
* -L<openblas_lib_path>：OpenBLAS 的函式庫路徑，例如 /usr/lib 或 /usr/lib/aarch64-linux-gnu。
* -lopenblas：連結 OpenBLAS 函式庫。

執行程式：
```bash
./run_demo
```
效能監測（另開終端）：
```bash
sudo tegrastats
```
