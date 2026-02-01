# TensorRT Python API 推論與 DLA 概念

### 題目說明
1. 使用 Practice 1 或 Practice 2 產生的 TensorRT 引擎
2. 透過 TensorRT Python API 載入引擎、配置緩衝記憶體、執行推論
3. 使用 `trtexec` 搭配 `--useDLACore=0 --allowGPUFallback` 模擬 DLA 編譯流程

> **注意**：Jetson Orin Nano 未搭載 DLA，本練習以 GPU 模擬 DLA 的操作流程，讓你了解 DLA 支援的運算子限制與 Fallback 機制。

### 作業練習
* 使用 `--dumpLayerInfo` 觀察層級部署資訊，找出哪些層被標記為 GPU Fallback
* 使用 Netron 開啟 ONNX 模型，對照 [DLA Supported Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay) 文件，分析哪些運算子不支援 DLA
