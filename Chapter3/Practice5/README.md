# INT8 量化精度驗證

### 前置條件
* 需要完成 Practice 1 的 SimpleCNN 模型定義

### 題目說明
1. 自訂 SimpleCNN 並匯出為 ONNX 格式
2. 分別編譯 FP32 與 INT8 兩個版本的 TensorRT 引擎
3. 使用 TensorRT Python API 載入兩個引擎，對相同測試資料進行推論

### 作業練習
* 比較 FP32 與 INT8 的推論輸出結果
* 計算兩者輸出的差異（使用 MSE 或最大絕對誤差），並分析 INT8 量化的適用情況
