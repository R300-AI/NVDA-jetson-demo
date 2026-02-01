# 匯出 YOLO 模型並比較 FP32 與 FP16 效能

### 題目說明

由於 NVIDIA 官方 PyTorch wheel 未包含 torchvision，而 ultralytics 套件依賴 torchvision，因此我們使用 **Google Colab** 匯出 YOLOv8 模型的 ONNX 檔案。

1. 在 Google Colab 執行 `export_yolov8.py` 匯出 yolov8n.onnx，並下載到 Jetson。
2. 使用 trtexec 將 ONNX 模型編譯成 FP32 及 FP16 兩個不同精度的 TensorRT 引擎。
3. 使用 trtexec 工具執行推論，並加上 --dumpProfile，列出每一層的執行時間與資源使用情況。

### 工作流程

**Step 1: 在 Google Colab 匯出 ONNX**

1. 開啟 [Google Colab](https://colab.research.google.com/)
2. 將 `export_yolov8.py` 的內容貼到 Colab 並執行
3. 下載產生的 `yolov8n.onnx` 檔案
4. 傳輸到 Jetson（使用 scp、USB 或其他方式）

**Step 2: 在 Jetson 上編譯 TensorRT 引擎**

```bash
# 編譯 FP32 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine

# 編譯 FP16 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
```

**Step 3: 執行效能分析**

```bash
trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile --exportProfile=yolov8n_fp32_profile.json
trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile --exportProfile=yolov8n_fp16_profile.json
```

### 作業練習
解析 profile 輸出，觀察 YOLO 模型在 FP32 與 FP16 模式下各層的執行時間分布與總推論效能差異（如：Conv 與 Detection Head 的耗時情況）。
