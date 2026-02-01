# TensorRT 模型部署與量化優化

這個學習資源旨在讓你掌握 **TensorRT** 的基本部署流程與量化技術。你將學會如何將 PyTorch 模型轉換為 ONNX 格式、使用 `trtexec` 編譯 TensorRT 引擎，並透過 `--dumpProfile` 與 `--dumpLayerInfo` 進行效能分析。

> **注意**：Jetson Orin Nano 未搭載 DLA (Deep Learning Accelerator)，Practice 3 將以 GPU 模擬 DLA 的操作流程，讓學生了解 DLA 部署的概念與限制。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例。

1. 查詢系統版本

    ```bash
    # 查看 JetPack 版本
    apt show nvidia-jetpack 2>/dev/null | grep Version

    # 查看 L4T 版本 (JetPack 6.2 對應 L4T R36.4)
    cat /etc/nv_tegra_release

    # 查看 Python 版本
    python3 --version
    ```

2. 確認 TensorRT 與 trtexec 工具

    ```bash
    # 確認 TensorRT 版本
    dpkg -l | grep tensorrt

    # 找到 trtexec 路徑
    find /usr -name "trtexec" 2>/dev/null
    # 輸出: /usr/src/tensorrt/bin/trtexec

    # 加入 PATH（建議加到 ~/.bashrc）
    export PATH=$PATH:/usr/src/tensorrt/bin
    ```

3. 安裝 Python 套件

    ```bash
    # libjpeg-turbo8-dev 是 libjpeg-dev 的實際提供者（JetPack 6.2 需使用此套件）
    sudo apt install -y python3-pip libopenblas-dev libjpeg-turbo8-dev zlib1g-dev libpng-dev
    pip3 install --upgrade pip
    ```

    ```
    # issue
    hunter@hunter-jeston:/usr/src/tensorrt/bin$ sudo apt install -y python3-pip libopenblas-dev libjpeg-turbo8-dev zlib1g-dev libpng-dev
正在讀取套件清單... 完成
正在重建相依關係... 完成  
正在讀取狀態資料... 完成  
E: 找不到套件 libjpeg-turbo8-dev

    ```
    

    **⚠️ 重要：Jetson 上必須使用 NVIDIA 官方 PyTorch wheel**

    ```bash
    # ❌ 錯誤方式 - 這樣安裝的是 CPU 版本，無法使用 GPU
    # pip3 install torch torchvision

    # ✅ Step 1: 安裝 NVIDIA 官方 PyTorch (JetPack 6.2 + Python 3.10)
    pip3 install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

    # ✅ Step 2: 從源碼編譯 torchvision（約需 10-20 分鐘）
    git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    python3 setup.py install --user
    cd ..

    # ✅ Step 3: 安裝其他套件（ultralytics 須用 --no-deps 避免覆蓋 torch）
    pip3 install pillow numpy onnx opencv-python
    pip3 install ultralytics --no-deps
    pip3 install py-cpuinfo psutil pyyaml tqdm requests
    ```

    > **說明**：`ultralytics` 會提示缺少 `polars`，可忽略（本教材不需要）。

    > 可用 wheel 列表：https://developer.download.nvidia.cn/compute/redist/jp/

    **驗證 GPU 支援**

    ```python
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")  # 應為 True
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ```

4. 安裝視覺化工具（工作站）

    於 **Workstation** 安裝 [Netron](https://netron.app/) 以視覺化 ONNX 模型結構。

## 編譯與執行

1. 匯出 ONNX 模型

    ```bash
    python3 <export_script>.py
    ```

2. 編譯 TensorRT 引擎

    ```bash
    # FP32 精度
    trtexec --onnx=<model>.onnx --saveEngine=<model>_fp32.engine

    # FP16 精度
    trtexec --onnx=<model>.onnx --saveEngine=<model>_fp16.engine --fp16

    # INT8 精度（需要校正資料）
    trtexec --onnx=<model>.onnx --saveEngine=<model>_int8.engine --int8 --calib=<calib>.cache
    ```

3. 執行推論與效能分析

    ```bash
    # 執行推論並輸出效能報告
    trtexec --loadEngine=<model>.engine --dumpProfile --dumpLayerInfo
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
| `--dumpLayerInfo` | 輸出層級部署資訊 |
| `--exportLayerInfo=<path>` | 將層資訊匯出為 JSON |
| `--verbose` | 顯示詳細編譯過程 |

## TensorRT 部署基礎

本節提供完成各 Practice 所需的核心概念。

### ONNX 模型匯出

PyTorch 模型可透過 `torch.onnx.export` 匯出為 ONNX 格式：

```python
import torch
import torchvision.models as models

model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=13)
```

### INT8 量化校正

INT8 量化需要校正資料來計算每層的動態範圍：

```python
import numpy as np

# 產生校正資料（實際應使用真實資料）
calib_data = np.random.randn(100, 3, 224, 224).astype(np.float32)
calib_data.tofile("calibration.bin")
```

```bash
trtexec --onnx=model.onnx --int8 --calib=calib.cache \
        --calibBatchSize=1 --saveEngine=model_int8.engine
```

### Profile 分析重點

執行 `--dumpProfile` 後，觀察以下指標：

- **Layer Time**：各層執行時間（ms）
- **Memory**：各層記憶體使用量
- **Total Time**：整體推論時間

