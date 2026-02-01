# Chapter3 練習題解答 - TensorRT 模型部署與量化優化

本資料夾包含 Chapter3 所有練習題的完整解答。

---

## 環境需求

請確保 JetPack 6.2 已安裝完成。

**⚠️ 重要：請參考 Chapter3/README.md 的環境設定說明安裝 PyTorch**

```bash
# NVIDIA 官方 PyTorch wheel (JetPack 6.2)
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

# 其他套件
pip3 install torchvision --no-deps
pip3 install pillow numpy onnx ultralytics
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
        --dumpProfile --dumpLayerInfo

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

## Lab 3: DLA 操作流程模擬 (GPU 版)

**檔案:** `lab3_dla_fallback.py`

### 說明

由於 Jetson Orin Nano 沒有搭載 DLA，本練習使用 GPU 來模擬 DLA 的操作流程，
讓學員了解 TensorRT 的 Layer 分析與 Fallback 機制。

### 執行步驟

```bash
# 1. 匯出包含 Layer 資訊的 ONNX 模型
python3 lab3_dla_fallback.py

# 2. 編譯引擎並分析 Layer 資訊
trtexec --onnx=mobilenet_v2.onnx --saveEngine=mobilenet_v2.engine \
        --fp16 --dumpLayerInfo --exportLayerInfo=layer_info.json

# 3. 觀察 Layer 資訊
cat layer_info.json
```

### 預期輸出
- `mobilenet_v2.onnx` - MobileNet V2 ONNX 模型
- `layer_info.json` - 各 Layer 的詳細資訊

---

## Lab 4: INT8 PTQ 校準資料生成

**檔案:** `lab4_generate_calib_data.py`

### 執行步驟

```bash
# 1. 生成校準資料
python3 lab4_generate_calib_data.py

# 2. 編譯 INT8 引擎 (需要校準資料)
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_int8.engine \
        --int8 --calib=calibration_data \
        --dumpProfile --dumpLayerInfo

# 3. 效能比較 (FP32 vs INT8)
trtexec --loadEngine=resnet50_fp32.engine --iterations=100 --dumpProfile
trtexec --loadEngine=resnet50_int8.engine --iterations=100 --dumpProfile
```

### 預期輸出
- `calibration_data/` - 校準資料目錄
- INT8 引擎約有 2-4x 加速 (視模型而定)

---

## Lab 5: ResNet18 QAT with CIFAR-10

**檔案:** `lab5_qat_resnet18_cifar10.py`

### 執行步驟

```bash
# 1. 執行 QAT 訓練並匯出 ONNX
python3 lab5_qat_resnet18_cifar10.py

# 2. 編譯 INT8 引擎
trtexec --onnx=resnet18_qat_cifar10.onnx --saveEngine=resnet18_qat.engine \
        --int8 --shapes=input:1x3x32x32 \
        --dumpProfile --dumpLayerInfo

# 3. 效能測試
trtexec --loadEngine=resnet18_qat.engine --iterations=100 --dumpProfile
```

### 預期輸出
- `resnet18_qat_cifar10.onnx` - QAT 量化後的 ONNX 模型
- 模型準確率約 70-80% (5 epochs)
- 更長訓練時間可達到更高準確率

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
