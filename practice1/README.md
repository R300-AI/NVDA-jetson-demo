# Jetson Orin 程式實作專案 - (1)

在這個學習資源中，使用者將學到如何使用 `gcc/g++` 編譯 `Eigen` 與 `OpenBLAS` 函式庫的程式碼，以此完成基本的矩陣與數值運算；並透過 `tegrastats` 工具觀察 CPU、GPU 與記憶體在執行過程的效能表現。

### 1. 環境準備

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
4. 檢查函式庫安裝路徑
  * Eigen
    ```
    ls /usr/include/eigen3
    ```
  * OpenBLAS
    ```
    ls /usr/lib/aarch64-linux-gnu | grep openblas
    ```
5. 執行效能監測工具
```
sudo tegrastats
```
會看到 CPU/GPU/記憶體的即時狀況，按 Ctrl+C 結束。

2. 編譯程式
3. 效能監測與執行

延伸練習題
