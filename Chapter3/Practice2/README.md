# 匯出 YOLOv8 模型並比較 FP32 與 FP16 效能

### 題目說明
1. 在 Google Colab 匯出 YOLOv8n 為 ONNX 格式，並下載到 Jetson
2. 使用 `trtexec` 分別編譯 FP32 與 FP16 兩個版本的 TensorRT 引擎
3. 載入引擎執行推論

> **說明**：由於 NVIDIA 官方 PyTorch wheel 未包含 torchvision，而 ultralytics 套件依賴 torchvision，因此使用 Google Colab 匯出 ONNX 檔案。

### 作業練習
* 使用 `--dumpProfile` 與 `--dumpLayerInfo` 分別分析 FP32 與 FP16 的效能
* 解析 profile 輸出，觀察 YOLO 模型在 FP32 與 FP16 模式下各層的執行時間分布
* 比較 FP32 與 FP16 的總推論效能差異（如：Conv 與 Detection Head 的耗時情況）
