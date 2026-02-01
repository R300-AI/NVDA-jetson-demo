# DLA Operator 驗證與 Fallback 練習

### 題目說明
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset=17)。
2. 使用 trtexec 工具將 ONNX 模型編譯成 INT8 精度的 TensorRT 引擎，並觀察層資訊。
3. 使用 --dumpLayerInfo 輸出每一層的部署資訊，模擬 DLA 部署流程。

> **注意**：Jetson Orin Nano 未搭載 DLA，本練習以 GPU 模擬 DLA 的操作流程，讓學生了解 DLA 支援的運算子限制與 Fallback 機制。

### 作業練習
使用 Netron 開啟 ONNX 模型，檢視模型結構，找出理論上不支援 DLA 的運算子 (Operators)。
