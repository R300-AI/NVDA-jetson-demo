# 透過 Polygraphy 實現 INT8 PTQ

### 題目說明
1. 使用 Practice 2 產生的 `yolov8n.onnx` 模型。
2. 建立 `data_loader.py` 腳本，從 `calib_images/` 資料夾載入真實校正圖片。
3. 使用 Polygraphy 工具進行 INT8 校正，產生 calibration cache 檔案。
4. 使用 trtexec 載入 cache 編譯 INT8 引擎，並加上 `--dumpProfile` 觀察效能。

> **提示**：你可以從 COCO 或 ImageNet 下載 100-500 張代表性圖片放入 `calib_images/` 資料夾。

### 作業練習
1. 完成 `data_loader.py`，從 `calib_images/` 載入真實圖片作為校正資料。
2. 比較 FP16 與 INT8 (PTQ) 的推論效能差異。
