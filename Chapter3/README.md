# TensorRT 模型部署與量化優化

這個學習資源旨在讓你掌握 **TensorRT** 的基本部署流程與量化技術。你將學會如何將 PyTorch 模型轉換為 ONNX 格式、使用 `trtexec` 編譯 TensorRT 引擎，並透過 `--dumpProfile` 與 `--dumpLayerInfo` 進行效能分析。

> **注意**：Jetson Orin Nano 未搭載 DLA (Deep Learning Accelerator)，Practice 3 將以 GPU 模擬 DLA 的操作流程，讓學生了解 DLA 部署的概念與限制。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例。

1. 確認 TensorRT 與 trtexec 工具

    ```bash
    # 確認 TensorRT 版本
    dpkg -l | grep tensorrt

    # 找到 trtexec 路徑，通常位於 /usr/src/tensorrt/bin/trtexec
    find /usr -name "trtexec" 2>/dev/null

    # 設定 trtexec 路徑 (加入 PATH)
    export PATH=$PATH:/usr/src/tensorrt/bin
    ```

2. 安裝 Python 套件

    ```bash
    sudo apt install -y python3-pip libopenblas-dev
    pip3 install --upgrade pip
    ```

    **⚠️ 重要：Jetson 上必須使用 NVIDIA 官方 PyTorch wheel**

    根據 [NVIDIA 官方文檔](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)，JetPack 6.2 對應 PyTorch 2.7.0/2.8.0。

    ```bash
    # ❌ 錯誤方式 - pip 安裝的是 CPU 版本，無法使用 GPU
    # pip3 install torch torchvision

    # ✅ 正確方式 - 使用 NVIDIA 官方 wheel (JetPack 6.2)
    # 方法一：使用 pip index URL (推薦)
    pip3 install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/

    # 方法二：直接下載 wheel 檔案
    # 先查看可用版本: https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/
    # wget https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/<wheel檔案名稱>
    # pip3 install <wheel檔案名稱>

    # 安裝其他套件
    pip3 install pillow numpy onnx ultralytics
    ```
    ```
    # result
    hunter@hunter-jeston:/usr/src/tensorrt/bin$ export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c
hunter@hunter-jeston:/usr/src/tensorrt/bin$ python3 -m pip install --no-cache $TORCH_INSTALL
Defaulting to user installation because normal site-packages is not writeable
Collecting https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c
  ERROR: HTTP error 404 while getting https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c
ERROR: Could not install requirement https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c because of HTTP error 404 Client Error: Not Found for url: https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c for URL https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/2.7.0a0+79aa17489c
hunter@hunter-jeston:/usr/src/tensorrt/bin$ 

    ```

    > 參考連結：
    > - [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
    > - [PyTorch for Jetson Release Notes](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)

    **驗證 GPU 支援**

    ```python
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ```

3. 安裝視覺化工具（工作站）

    於 **Workstation** 安裝 [**Netron**](https://netron.app/) 以視覺化 ONNX 模型結構。

## 編譯與執行流程

### Step 1: 匯出 ONNX 模型

```bash
python3 export_model.py
```

### Step 2: 編譯 TensorRT 引擎

```bash
# FP32 精度
trtexec --onnx=model.onnx --saveEngine=model_fp32.engine

# FP16 精度
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# INT8 精度（需要校正資料）
trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 --calib=calib.cache
```

### Step 3: 執行推論與效能分析

```bash
# 執行推論並輸出效能報告
trtexec --loadEngine=model.engine --dumpProfile --exportProfile=profile.json

# 輸出層級部署資訊
trtexec --loadEngine=model.engine --dumpLayerInfo --exportLayerInfo=layers.json
```

## trtexec 常用參數

| 參數 | 說明 |
|------|------|
| `--onnx=<path>` | 指定 ONNX 模型路徑 |
| `--saveEngine=<path>` | 儲存編譯後的 TensorRT 引擎 |
| `--loadEngine=<path>` | 載入已編譯的 TensorRT 引擎 |
| `--fp16` | 啟用 FP16 精度 |
| `--int8` | 啟用 INT8 精度 |
| `--calib=<path>` | 指定 INT8 校正快取檔案 |
| `--shapes=<spec>` | 指定輸入形狀，例如 `input:1x3x224x224` |
| `--dumpProfile` | 輸出每層執行時間 |
| `--dumpLayerInfo` | 輸出層級部署資訊（GPU/DLA 分配）|
| `--exportProfile=<path>` | 將 Profile 匯出為 JSON |
| `--exportLayerInfo=<path>` | 將層資訊匯出為 JSON |
| `--verbose` | 顯示詳細編譯過程 |
| `--useDLACore=<N>` | 指定使用 DLA 核心（0 或 1）|
| `--allowGPUFallback` | 允許不支援 DLA 的層回退至 GPU |

## TensorRT 部署基礎

### ONNX 模型匯出

PyTorch 模型可透過 `torch.onnx.export` 匯出為 ONNX 格式：

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=13)
```

### INT8 量化校正

INT8 量化需要校正資料來計算每層的動態範圍：

```bash
# 使用 trtexec 進行 INT8 校正
trtexec --onnx=model.onnx --int8 --calib=calib.cache \
        --calibrationData=calib.bin --calibrationBatchSize=1
```

### Profile 分析重點

執行 `--dumpProfile` 後，觀察以下指標：

- **Layer Time**：各層執行時間（ms）
- **Memory**：各層記憶體使用量
- **Total Time**：整體推論時間

