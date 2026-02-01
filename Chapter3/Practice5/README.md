# INT8 量化精度驗證

### 題目說明
1. 使用 Practice 2 編譯的 FP32 引擎（`yolov8n_fp32.engine`）與 Practice 4 編譯的 INT8 引擎（`yolov8n_int8.engine`）
2. 使用 TensorRT Python API 載入兩個引擎，對相同測試資料進行推論
3. 比較 FP32 與 INT8 的輸出差異，驗證量化精度

### 作業練習
* 計算 FP32 與 INT8 輸出的 MSE（均方誤差）與最大絕對誤差
* 分析 INT8 量化對 YOLOv8 模型精度的影響
