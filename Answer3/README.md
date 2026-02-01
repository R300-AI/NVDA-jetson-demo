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
pip3 install pillow numpy onnx opencv-python
pip3 install timm  # 用於載入預訓練模型
pip3 install ultralytics --no-deps
pip3 install py-cpuinfo psutil pyyaml tqdm requests
pip3 install polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

---

## Lab 1: ResNet-50 ONNX 匯出與 TensorRT 推論

**檔案:** `lab1_export_resnet50.py`

### 執行步驟

```bash
# 1. 匯出 ONNX 模型
python3 lab1_export_resnet50.py

# 2. 編譯 TensorRT FP32 引擎
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp32.engine \
        --shapes=input:1x3x224x224 --dumpProfile --dumpLayerInfo

# 3. 執行推論效能測試
trtexec --loadEngine=resnet50_fp32.engine --iterations=100 \
        --dumpProfile
```

### 預期輸出
- `resnet50.onnx` - ONNX 格式的 ResNet-50 模型
- `resnet50_fp32.engine` - TensorRT FP32 引擎

---

## Lab 2: YOLOv8 FP32 vs FP16 效能比較

**檔案:** `lab2_export_yolov8.py`

### 執行步驟

```bash
# 1. 匯出 YOLOv8 ONNX 模型
python3 lab2_export_yolov8.py

# 2. 編譯 FP32 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine \
        --dumpProfile --dumpLayerInfo

# 3. 編譯 FP16 引擎
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine \
        --fp16 --dumpProfile --dumpLayerInfo

# 4. 執行效能比較
trtexec --loadEngine=yolov8n_fp32.engine --iterations=100 --dumpProfile
trtexec --loadEngine=yolov8n_fp16.engine --iterations=100 --dumpProfile
```

### 預期輸出
- `yolov8n.onnx` - YOLOv8n ONNX 模型
- 效能報告顯示 FP16 約有 2x 加速

---

## Lab 3: TensorRT Python API 推論與 DLA 概念

**檔案:** `lab3_trt_inference.py`

### 說明

使用 TensorRT Python API 載入引擎並執行推論。由於 Jetson Orin Nano 沒有搭載 DLA，
本練習以 GPU 模擬 DLA 的操作流程，讓你了解 DLA 支援的運算子限制與 Fallback 機制。

### 執行步驟

```bash
# 1. 確認已有 Practice 1 產生的 engine 檔案
ls resnet50_fp32.engine

# 2. 執行 TensorRT Python API 推論
python3 lab3_trt_inference.py

# 3. 觀察 Layer 資訊（模擬 DLA 部署）
trtexec --loadEngine=resnet50_fp32.engine \
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

## Lab 4: INT8 PTQ 校正（使用真實圖片）

**檔案:** `lab4_data_loader.py`

### 說明

使用 Polygraphy 搭配真實校正圖片進行 INT8 校正。`data_loader.py` 會從 `calib_images/` 
資料夾載入圖片作為校正資料。

### 執行步驟

```bash
# 1. 準備校正圖片（建議 100-500 張）
mkdir calib_images
# 從 COCO 或 ImageNet 下載代表性圖片

# 2. 使用 Polygraphy 產生 calibration cache
polygraphy convert yolov8n.onnx --int8 \
    --data-loader-script ./lab4_data_loader.py \
    --calibration-cache yolov8n_calib.cache

# 3. 使用 trtexec 編譯 INT8 引擎
trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \
        --saveEngine=yolov8n_int8.engine

# 4. 效能比較 (FP16 vs INT8)
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

使用 CIFAR-10 測試集驗證 FP32 與 INT8 引擎的精度差異，分析量化對精度的影響。

### 執行步驟

```bash
# 1. 匯出 ONNX 模型
python3 lab5_validate_int8.py --export

# 2. 編譯 FP32 引擎
trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_fp32.engine \
        --shapes=input:1x3x32x32

# 3. 編譯 INT8 引擎
trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_int8.engine \
        --shapes=input:1x3x32x32 --int8

# 4. 執行精度驗證
python3 lab5_validate_int8.py --validate
```

### 預期輸出
- `resnet18_cifar10.onnx` - ONNX 模型
- `resnet18_fp32.engine` - FP32 引擎
- `resnet18_int8.engine` - INT8 引擎
- 精度比較報告（FP32 vs INT8 Top-1 準確率）

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
