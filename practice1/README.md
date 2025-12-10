# Jetson Orin 程式實作專案 - (1)

本文件提供操作步驟，重點在四個工具：`gcc/g++`（編譯器，用來把程式碼轉成可執行檔）、`Eigen`（線性代數函式庫，處理小矩陣運算）、`OpenBLAS`（高效能數值庫，處理大矩陣運算）、`tegrastats`（效能監測工具，顯示 CPU/GPU/記憶體使用狀況）。請依照下列指引完成環境檢查、程式編譯與執行、效能觀察，最後在延伸練習題中進行數據紀錄與分析。


## 1. 環境準備

先更新套件清單：
```bash
sudo apt update
```
再裝基本工具：
```bash
sudo apt install -y build-essential libeigen3-dev libopenblas-dev
```
確認一下版本：
```bash
gcc --version
g++ --version
```
看一下 Eigen 路徑：
```
ls /usr/include/eigen3
```
檢查 OpenBLAS：
```
ls /usr/lib/aarch64-linux-gnu | grep openblas
```
最後跑一下 tegrastats：
```
sudo tegrastats
```
會看到 CPU/GPU/記憶體的即時狀況，按 Ctrl+C 結束。

2. 編譯程式
3. 效能監測與執行

延伸練習題
