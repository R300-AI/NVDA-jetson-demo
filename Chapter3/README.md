# TensorRT 模型部署與量化優化

這個學習資源旨在讓你掌握 **TensorRT** 的基本部署流程與量化技術。你將學會如何將 PyTorch 模型轉換為 ONNX 格式、使用 `trtexec` 編譯 TensorRT 引擎，並透過 `--dumpProfile` 與 `--dumpLayerInfo` 進行效能分析。

> **注意**：Jetson Orin Nano 未搭載 DLA (Deep Learning Accelerator)，Practice 3 將以 GPU 模擬 DLA 的操作流程，以此釐清 DLA 部署的概念與限制。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例。

1. 確認 TensorRT 與 trtexec 工具

    ```bash
    # 確認 TensorRT 版本
    dpkg -l | grep tensorrt

    # 找到 trtexec 路徑
    find /usr -name "trtexec" 2>/dev/null

    # 將輸出的路徑加入 PATH（通常為/usr/src/tensorrt/bin/trtexec）
    export PATH=$PATH:/usr/src/tensorrt/bin
    ```

2. 安裝基礎套件

    ```bash
    sudo apt-get install -y python3-pip libopenblas-dev
    pip3 install --upgrade pip
    ```

3. 安裝 PyTorch（NVIDIA 官方 wheel）

    ```bash
    # wheel 列表：https://developer.download.nvidia.cn/compute/redist/jp/
    pip3 install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    ```

4. 驗證安裝

    ```python
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")  # 應為 True
    ```

5. 安裝其他套件

    ```bash
    pip3 install "numpy<2" pillow onnx opencv-python pycuda
    pip3 install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
    ```


## 編譯與執行

1. 執行編譯指令

    ```bash
    trtexec --onnx=<model>.onnx --saveEngine=<model>.engine
    ```

    | 參數 | 說明 |
    |------|------|
    | `--onnx=<model>.onnx` | 輸入的 ONNX 模型路徑 |
    | `--saveEngine=<model>.engine` | 輸出的 TensorRT 引擎路徑 |
    | `--fp16` | 啟用 FP16 精度（速度與精確度平衡） |
    | `--int8` | 啟用 INT8 精度（僅供測試，使用隨機權重） |
    | `--calib=<name>.cache` | 指定校正快取檔案（搭配 `--int8` 使用） |

2. 執行推論效能分析

    ```bash
    trtexec --loadEngine=<model>.engine --dumpProfile --dumpLayerInfo
    ```

    | 參數 | 說明 |
    |------|------|
    | `--loadEngine=<model>.engine` | 載入已編譯的 TensorRT 引擎 |
    | `--dumpProfile` | 輸出每層執行時間 |
    | `--dumpLayerInfo` | 輸出層級部署資訊 |
    | `--exportLayerInfo=<name>.json` | 將層資訊匯出至 JSON 檔案（可選） |

## TensorRT 部署基礎知識

本節提供完成各 Practice 所需的核心概念。

### 匯出自訂 ONNX 模型

JetPack 6.2 的 TensorRT 10.x 支援 ONNX opset 9-20，建議使用 **opset 17**。

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 在此定義模型的層結構（如 Conv2d、Linear、ReLU 等）
    
    def forward(self, x):
        # 在此定義前向傳播的計算流程，決定資料如何經過各層
        return x

# 匯出 ONNX
model = MyModel()
model.eval()

batch_size, channels, height, width = 1, 3, 224, 224
dummy_input = torch.randn(batch_size, channels, height, width)

torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17, input_names=['input'], output_names=['output'])
```

### 匯出Ultralytics YOLOs 模型

請在 [Google Colab](https://colab.research.google.com/?hl=zh-tw) 執行以下程式碼匯出 YOLOv8 模型，再下載到 Jetson：

```python
!pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format='onnx', opset=17, imgsz=640, simplify=True)
```

## TensorRT Python API 推論

使用 Python 載入編譯好的 `.engine` 進行推論，步驟如下：

**步驟一**：載入引擎

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("model.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
```

**步驟二**：配置緩衝記憶體

```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Host = CPU, Device = GPU
h_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
h_output = np.empty((1, 1000), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
```

**步驟三**：執行推論

```python
cuda.memcpy_htod(d_input, h_input)       # CPU → GPU
context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(h_output, d_output)     # GPU → CPU
```

### INT8 模型量化與校正

INT8 量化需要校正資料來決定每層的量化範圍。步驟如下：

**步驟一**：建立 `data_loader.py`，定義 `load_data()` 函數產生校正樣本

```python
import numpy as np

def load_data():
    for _ in range(100):  # 校正樣本數量（建議 100~500）
        yield {"input": np.random.rand(1, 3, 224, 224).astype(np.float32)}
        # ↑ key 名稱須與 ONNX 模型的輸入名稱一致（可用 Netron 查看）
        # ↑ 形狀須與模型輸入一致
```

**步驟二**：執行 Polygraphy 校正，產生 `.cache` 與 INT8 引擎

```bash
polygraphy convert <model>.onnx --int8 \
    --data-loader-script ./data_loader.py \
    --calibration-cache <calib_name>.cache \
    -o <model>_int8.engine
```

> **提示**：產生的 `.cache` 可重複使用，後續可直接用 `trtexec --calib=model_calib.cache` 編譯。