# Jetson Orin 程式實作專案 - (1)

這份教學會一步一步帶你從環境準備、C++的編譯與執行，到利用 tegrastats 監測效能，最後再完成幾個變化題，練習操作流程。

---

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
