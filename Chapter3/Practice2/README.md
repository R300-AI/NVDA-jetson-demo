# 匯出 YOLO 模型並比較 FP32 與 FP16 效能

### 題目說明
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset=17)。
2. 使用 trtexec 將 ONNX 模型編譯成 FP32 及 FP16 兩個不同精度的 TensorRT 引擎 (--shapes=images:1x3x640x640)。
3. 使用 trtexec 工具執行推論，並加上 --dumpProfile，列出每一層的執行時間與資源使用情況。

### 作業練習
解析 profile 輸出，觀察 YOLO 模型在 FP32 與 FP16 模式下各層的執行時間分布與總推論效能差異（如：Conv 與 Detection Head 的耗時情況）。
