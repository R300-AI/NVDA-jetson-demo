# TensorRT 模型部署與量化優化

這個學習資源旨在讓你掌握 **TensorRT** 的基本部署流程與量化技術。你將學會如何將 PyTorch 模型轉換為 ONNX 格式、使用 `trtexec` 編譯 TensorRT 引擎，並透過 `--dumpProfile` 與 `--dumpLayerInfo` 進行效能分析。

> **注意**：Jetson Orin Nano 未搭載 DLA (Deep Learning Accelerator)，Practice 3 將以 GPU 模擬 DLA 的操作流程，讓學生了解 DLA 部署的概念與限制。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例。

1. 確認 TensorRT 與 trtexec 工具

    ```bash
    # 確認 TensorRT 版本
    dpkg -l | grep tensorrt

    # 找到 trtexec 路徑
    find /usr -name "trtexec" 2>/dev/null
    # 輸出: /usr/src/tensorrt/bin/trtexec

    # 加入 PATH（建議加到 ~/.bashrc）
    export PATH=$PATH:/usr/src/tensorrt/bin
    ```

2. 安裝基礎套件

    ```bash
    sudo apt-get install -y python3-pip libopenblas-dev
    pip3 install --upgrade pip
    pip3 install "numpy<2"  # PyTorch wheel 需要 NumPy 1.x
    ```

3. 安裝 NVIDIA 官方 PyTorch (JetPack 6.2 + Python 3.10)

    ```bash
    # wheel 列表：https://developer.download.nvidia.cn/compute/redist/jp/
    pip3 install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    ```

4. 安裝模型匯出與推論相關套件

    ```bash
    pip3 install pillow onnx opencv-python
    pip3 install timm
    pip3 install ultralytics --no-deps
    pip3 install py-cpuinfo psutil pyyaml tqdm requests
    pip3 install pycuda
    pip3 install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
    ```

5. 驗證安裝

    ```python
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")  # 應為 True
    ```

## 編譯與執行

1. 執行編譯指令

```bash
trtexec --onnx=<model>.onnx --saveEngine=<model>.engine
```

| 參數 | 說明 |
|------|------|
| `--onnx` | 輸入的 ONNX 模型路徑 |
| `--saveEngine` | 輸出的 TensorRT 引擎路徑 |
| `--fp16` | 啟用 FP16 精度（速度與精確度平衡） |
| `--int8` | 啟用 INT8 精度（僅供測試，使用隨機權重） |
| `--calib` | 指定已有的 calibration cache 檔案 (可選) |

2. 執行推論效能分析

```bash
trtexec --loadEngine=<model>.engine --dumpProfile --dumpLayerInfo
```

| 參數 | 說明 |
|------|------|
| `--loadEngine` | 載入已編譯的 TensorRT 引擎 |
| `--dumpProfile` | 輸出每層執行時間 |
| `--dumpLayerInfo` | 輸出層級部署資訊 |
| `--exportLayerInfo` | 將層資訊匯出至 JSON 檔案 |

## TensorRT API部署基礎

### ONNX 模型匯出

JetPack 6.2 的 TensorRT 10.x 支援 ONNX opset 9-20，建議使用 **opset 17**。

#### timm 預訓練模型

```python
import torch
import timm

# 可用模型請參考：https://huggingface.co/timm
model = timm.create_model('resnet50', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=17)
```

#### Ultralytics YOLOs

```python
from ultralytics import YOLO

# 可用模型請參考：https://docs.ultralytics.com/models/
model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=17)
```

### TensorRT Python API 推論

你可以使用 Python 直接載入編譯好的 `.engine` 進行推論。

#### 步驟一：載入 TensorRT 引擎

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("model.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
```

#### 步驟二：準備輸入輸出緩衝區

```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 定義形狀
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)  # ResNet50 輸出 1000 類別

# Host 記憶體（CPU）
h_input = np.random.randn(*input_shape).astype(np.float32)
h_output = np.empty(output_shape, dtype=np.float32)

# Device 記憶體（GPU）
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
```

#### 步驟三：執行推論

```python
# 複製輸入到 GPU
cuda.memcpy_htod(d_input, h_input)

# 執行推論
context.execute_v2([int(d_input), int(d_output)])

# 複製輸出回 CPU
cuda.memcpy_dtoh(h_output, d_output)

print(f"預測類別: {np.argmax(h_output)}")
```

### INT8 量化校正

`trtexec --int8` 本身不會執行真正的校正（使用隨機權重），你需要先透過 **Polygraphy** 產生 calibration cache，再以 trtexec 編譯引擎。

#### 步驟一：建立校正資料載入器

建立 `data_loader.py`，提供真實的校正資料：

```python
import numpy as np
from PIL import Image
import os

def load_data():
    """產生校正資料批次"""
    calib_dir = "./calib_images"  # 放置 100-500 張代表性圖片
    
    for img_name in os.listdir(calib_dir)[:100]:
        img = Image.open(os.path.join(calib_dir, img_name)).resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)   # 加入 batch 維度
        
        yield {"input": img_array}  # key 須與 ONNX 輸入名稱一致
```

#### 步驟二：使用 Polygraphy 產生 Calibration Cache

```bash
polygraphy convert model.onnx --int8 \
    --data-loader-script ./data_loader.py \
    --calibration-cache calib.cache
```

#### 步驟三：使用 trtexec 編譯 INT8 引擎

```bash
trtexec --onnx=model.onnx --int8 --calib=calib.cache --saveEngine=model_int8.engine
```

#### INT8 注意事項

| 項目 | 說明 |
|------|------|
| 校正資料量 | 建議 100-500 張具代表性的真實資料 |
| 精度差異 | INT8 可能造成 1-3% 精度下降，需驗證推論結果 |
| 不適用情況 | 輸出範圍大或分布不均的模型可能不適合 INT8 |
| 驗證方式 | 比較 FP32 與 INT8 輸出的 Top-1/Top-5 準確率 |

### DLA 部署與 GPU Fallback

DLA (Deep Learning Accelerator) 是 NVIDIA Jetson 系列部分裝置（如 Orin NX、AGX Orin）搭載的專用推論加速器。由於 DLA 僅支援部分運算子，不支援的層會自動 Fallback 到 GPU 執行。

> **注意**：Jetson Orin Nano 未搭載 DLA，但你可以透過 `--dumpLayerInfo` 了解哪些層理論上支援 DLA，為日後使用其他 Jetson 裝置做準備。

#### DLA 支援的運算子

| 類型 | 運算子 |
|------|--------|
| 卷積 | Conv, ConvTranspose |
| 全連接 | Gemm (Fully Connected) |
| 池化 | MaxPool, AveragePool |
| 激活 | ReLU, Sigmoid, Tanh |
| 正規化 | BatchNormalization, Scale |
| 元素運算 | Add, Sub, Mul, Max, Min |
| 其他 | Concatenation |

#### DLA 不支援的運算子（會 Fallback 到 GPU）

| 運算子 | 說明 |
|--------|------|
| Softmax | 常見於分類輸出層 |
| Resize / Upsample | 常見於物件偵測模型 |
| Split | 張量分割 |
| ReduceMean, ReduceMax | 聚合運算 |
| 特定 Pad 模式 | 部分填充模式不支援 |

#### 編譯指令（適用於有 DLA 的裝置）

```bash
trtexec --onnx=model.onnx --saveEngine=model_dla.engine \
        --useDLACore=0 --allowGPUFallback \
        --int8 --fp16 \
        --dumpLayerInfo --exportLayerInfo=layers.json
```

| 參數 | 說明 |
|------|------|
| `--useDLACore=0` | 使用第一個 DLA 核心 |
| `--allowGPUFallback` | 允許不支援的層 Fallback 到 GPU |

### Profile 分析重點

執行 `--dumpProfile` 後，觀察以下指標：

| 指標 | 說明 |
|------|------|
| Layer Time | 各層執行時間（ms） |
| Memory | 各層記憶體使用量 |
| Total Time | 整體推論時間 |
| Precision | 各層實際使用的精度（FP32/FP16/INT8） |

