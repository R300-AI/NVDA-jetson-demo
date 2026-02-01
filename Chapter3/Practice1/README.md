# TensorRT 基本部署流程

### 題目說明
1. 自訂一個簡單的 CNN 分類器並匯出為 ONNX 格式
2. 使用 `trtexec` 將 ONNX 編譯成 FP32 的 TensorRT 引擎
3. 載入引擎執行推論

### 作業練習
* 使用 `--dumpProfile` 與 `--dumpLayerInfo` 分析效能
* 解析 profile 輸出，觀察各層的執行時間分布，找出耗時最多的層並分析其原因
