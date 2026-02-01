# 透過 Polygraphy 實現 INT8 PTQ

### 前置條件
* 需要 Practice 2 產生的 YOLOv8 ONNX 模型

### 題目說明
1. 建立 `data_loader.py`，定義 `load_data()` 函數產生校正資料
2. 使用 `polygraphy convert` 搭配 `--int8` 與 `--data-loader-script` 進行校正
3. 產生 calibration cache 與 INT8 引擎

> **提示**：本練習使用隨機數據作為示範，實際應用中應使用真實的代表性資料進行校正。

### 作業練習
* 使用 `--dumpProfile` 與 `--dumpLayerInfo` 觀察 INT8 引擎效能
* 比較 Practice 2 的 FP16 引擎與本練習 INT8 引擎的推論效能差異
