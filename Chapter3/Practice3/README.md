# TensorRT Python API 推論與 DLA 概念

### 題目說明
1. 使用 Practice 1 或 Practice 2 產生的 `.engine` 檔案。
2. 使用 TensorRT Python API 載入引擎並執行推論。
3. 使用 `--dumpLayerInfo` 觀察層級部署資訊，了解 DLA 支援的運算子與 GPU Fallback 機制。

> **注意**：Jetson Orin Nano 未搭載 DLA，本練習以 GPU 模擬 DLA 的操作流程，讓你了解 DLA 支援的運算子限制與 Fallback 機制。日後使用 Orin NX 或 AGX Orin 時，可直接套用相同流程。

### 作業練習
1. 完成 `trt_inference.py`，使用 TensorRT Python API 載入 `.engine` 並執行推論。
2. 使用 Netron 開啟 ONNX 模型，檢視模型結構，找出理論上不支援 DLA 的運算子。
