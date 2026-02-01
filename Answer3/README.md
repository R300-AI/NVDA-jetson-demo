# Chapter3 練習題解答 - TensorRT 模型部署與量化優化

本資料夾包含 Chapter3 所有練習題的完整解答。

---

## 環境需求

請確保 JetPack 6.2 已安裝完成。

**⚠️ 重要：請參考 Chapter3/README.md 的環境設定說明安裝 PyTorch**

```bash
# NVIDIA 官方 PyTorch wheel (JetPack 6.2)
pip3 install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 其他套件
pip3 install "numpy<2" pillow onnx opencv-python pycuda
pip3 install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

> **注意**：由於 NVIDIA 官方 PyTorch wheel 未包含 torchvision，而 ultralytics 套件依賴 torchvision，因此 YOLOv8 模型的 ONNX 匯出將透過 **Google Colab** 完成。

---

## Lab 1: SimpleCNN ONNX 匯出與 TensorRT 推論

**檔案:** `lab1_export_simple_cnn.py`

### 執行步驟

```bash
# 1. 匯出 ONNX 模型
python3 lab1_export_simple_cnn.py

# 2. 編譯 TensorRT FP32 引擎
trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine \
        --dumpProfile --dumpLayerInfo

# 3. 執行推論效能測試
trtexec --loadEngine=simple_cnn_fp32.engine --iterations=100 \
        --dumpProfile
```

### 預期輸出
- `simple_cnn.onnx` - ONNX 格式的 SimpleCNN 模型
- `simple_cnn_fp32.engine` - TensorRT FP32 引擎

---

## Lab 2: YOLOv8 FP32 vs FP16 效能比較（使用 Google Colab 匯出）

**檔案:** `lab2_export_yolov8.py`

### 說明

由於 NVIDIA 官方 PyTorch wheel 未包含 torchvision，而 ultralytics 套件依賴 torchvision，
因此我們使用 **Google Colab** 匯出 YOLOv8 模型的 ONNX 檔案，再下載到 Jetson 進行 TensorRT 編譯。

### 執行步驟

**Step 1: 在 Google Colab 匯出 ONNX**

1. 開啟 [Google Colab](https://colab.research.google.com/)
2. 新增一個 Code Cell，貼上以下程式碼：

```python
# 安裝 ultralytics
!pip install ultralytics

from ultralytics import YOLO

# 載入預訓練模型
model = YOLO('yolov8n.pt')

# 匯出 ONNX 模型
model.export(
    format='onnx',
    opset=17,
    imgsz=640,
    simplify=True
)

# 下載 ONNX 檔案
from google.colab import files
files.download('yolov8n.onnx')
```

3. 執行後下載 `yolov8n.onnx`
4. 傳輸到 Jetson（使用 scp、USB 或其他方式）

**Step 2: 在 Jetson 上編譯與測試**

```bash
# 編譯 FP32 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine \
        --dumpProfile --dumpLayerInfo

# 編譯 FP16 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine \
        --fp16 --dumpProfile --dumpLayerInfo

# 執行效能比較
trtexec --loadEngine=yolov8n_fp32.engine --iterations=100 --dumpProfile
trtexec --loadEngine=yolov8n_fp16.engine --iterations=100 --dumpProfile
```

### 預期輸出
- `yolov8n.onnx` - YOLOv8n ONNX 模型（從 Colab 下載）
- `yolov8n_fp32.engine` - FP32 TensorRT 引擎
- `yolov8n_fp16.engine` - FP16 TensorRT 引擎
- 效能報告顯示 FP16 約有 2x 加速

---

## Lab 3: TensorRT Python API 推論與 DLA 概念

**檔案:** `lab3_trt_inference.py`

### 說明

使用 TensorRT Python API 載入引擎並執行推論。由於 Jetson Orin Nano 沒有搭載 DLA，
本練習以 GPU 模擬 DLA 的操作流程，讓你了解 DLA 支援的運算子限制與 Fallback 機制。

### 執行步驟

```bash
# 1. 確認已有 Lab 1 產生的 engine 檔案
ls simple_cnn_fp32.engine

# 2. 執行 TensorRT Python API 推論
python3 lab3_trt_inference.py

# 3. 觀察 Layer 資訊（模擬 DLA 部署）
trtexec --loadEngine=simple_cnn_fp32.engine \
        --dumpLayerInfo --exportLayerInfo=layers.json
```

### DLA 部署指令（適用於 Orin NX / AGX Orin）

```bash
trtexec --onnx=model.onnx --saveEngine=model_dla.engine \
        --useDLACore=0 --allowGPUFallback \
        --int8 --fp16 --dumpLayerInfo
```

### 預期輸出
- 成功載入引擎並執行推論
- 顯示預測類別與信心分數
- `layers.json` - 各 Layer 的詳細資訊

---

## Lab 4: INT8 PTQ 校正（使用隨機數據示範）

**檔案:** `lab4_data_loader.py`

### 說明

使用 Polygraphy 進行 INT8 校正。本範例使用隨機數據作為示範，實際應用中應使用真實的代表性資料。

### 執行步驟

```bash
# 1. 使用 Polygraphy 產生 calibration cache
polygraphy convert yolov8n.onnx --int8 \
    --data-loader-script ./lab4_data_loader.py \
    --calibration-cache yolov8n_calib.cache

# 2. 使用 trtexec 編譯 INT8 引擎
trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \
        --saveEngine=yolov8n_int8.engine

# 3. 效能比較 (FP16 vs INT8)
trtexec --loadEngine=yolov8n_fp16.engine --iterations=100 --dumpProfile
trtexec --loadEngine=yolov8n_int8.engine --iterations=100 --dumpProfile
```

### 預期輸出
- `yolov8n_calib.cache` - 校正快取檔案
- `yolov8n_int8.engine` - INT8 引擎
- INT8 引擎約有 1.5-2x 加速（相較於 FP16）

---

## Lab 5: INT8 量化精度驗證

**檔案:** `lab5_validate_int8.py`

### 說明

使用隨機測試資料驗證 FP32 與 INT8 引擎的輸出差異，分析量化對精度的影響。

### 執行步驟

```bash
# 1. 匯出 ONNX 模型
python3 lab5_validate_int8.py --export

# 2. 編譯 FP32 引擎
trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine

# 3. 編譯 INT8 引擎
trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8

# 4. 執行精度驗證
python3 lab5_validate_int8.py --validate
```

### 預期輸出
- `simple_cnn.onnx` - ONNX 模型
- `simple_cnn_fp32.engine` - FP32 引擎
- `simple_cnn_int8.engine` - INT8 引擎
- 精度比較報告（MSE、最大絕對誤差、預測一致性）

---

## 效能比較總結

執行完所有練習後，可以得到以下效能比較：

| 精度 | 引擎大小 | 推論延遲 | 相對加速 |
|------|---------|---------|---------|
| FP32 | ~100MB  | baseline | 1.0x |
| FP16 | ~50MB   | ~50%     | ~2.0x |
| INT8 | ~25MB   | ~25%     | ~4.0x |

*實際數據視模型與硬體而異*

---

## 故障排除

### 1. CUDA Out of Memory
```bash
# 減少 batch size
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x224x224
```

### 2. ONNX 匯出失敗
```bash
# 確認 torch 版本
pip3 show torch
# 建議使用 torch >= 2.0
```

### 3. TensorRT 編譯失敗
```bash
# 檢查 ONNX 模型
polygraphy inspect model model.onnx
```

---

## 參考資料

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
