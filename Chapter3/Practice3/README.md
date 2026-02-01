# TensorRT Python API 推論與 DLA 概念

### 題目說明
1. 使用 Practice 1 或 Practice 2 產生的 TensorRT 引擎
2. 透過 TensorRT Python API 載入引擎、配置緩衝記憶體、執行推論
3. 了解 DLA（Deep Learning Accelerator）的概念與 `--useDLACore`、`--allowGPUFallback` 參數用途

> **注意**：Jetson Orin Nano **未搭載 DLA 硬體**，執行 `--useDLACore=0` 會出現 `Cannot create DLA engine` 錯誤，這是預期中的行為。

### 作業練習
* 使用 `--dumpLayerInfo` 觀察 GPU 模式下的層級部署資訊
* 閱讀 [DLA Supported Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay) 文件，了解運算子限制，並撰寫 Python 腳本自動檢查 DLA 運算子支援情況
* 使用 [Netron](https://netron.app/) 開啟 ONNX 模型，對照 DLA 文件找出不支援的運算子位置

