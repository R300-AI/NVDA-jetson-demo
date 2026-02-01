# 透過 Polygraphy 實現 INT8 PTQ

### 題目說明
1. 使用 Practice 2 產生的 `yolov8n.onnx` 模型。
2. 建立 `data_loader.py` 腳本，產生隨機數據作為校正資料（示範用途）。
3. 使用 Polygraphy 工具進行 INT8 校正，產生 calibration cache 檔案。
4. 使用 trtexec 載入 cache 編譯 INT8 引擎，並加上 `--dumpProfile` 觀察效能。

> **提示**：本練習使用隨機數據作為示範，實際應用中應使用真實的代表性資料進行校正。

### 作業練習
1. 完成 `data_loader.py`，產生隨機數據作為校正資料。
2. 比較 FP16 與 INT8 (PTQ) 的推論效能差異。
