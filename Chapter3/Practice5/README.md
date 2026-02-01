# INT8 量化精度驗證

### 題目說明
1. 使用純 PyTorch 自訂 SimpleCNN 並匯出為 ONNX 格式。
2. 分別編譯 FP32 與 INT8 兩個版本的 TensorRT 引擎。
3. 使用 TensorRT Python API 載入兩個引擎，對隨機測試資料進行推論。
4. 比較 FP32 與 INT8 的輸出差異，分析量化對精度的影響。

### 作業練習
* 完成 `validate_int8.py`，比較 FP32 與 INT8 的推論結果。
* 計算兩者輸出的差異（使用 MSE 或最大絕對誤差），並分析 INT8 量化的適用情況。
