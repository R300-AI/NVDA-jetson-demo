# INT8 量化精度驗證

### 題目說明
1. 使用 `download_cifar10()` 下載 CIFAR-10 測試集作為驗證資料。
2. 使用 torchvision 建立 ResNet18 模型並匯出為 ONNX 格式。
3. 分別編譯 FP32 與 INT8 兩個版本的 TensorRT 引擎。
4. 使用 TensorRT Python API 載入兩個引擎，對 CIFAR-10 測試集進行推論。
5. 比較 FP32 與 INT8 的 Top-1 準確率，分析量化對精度的影響。

### 作業練習
1. 完成 `validate_int8.py`，比較 FP32 與 INT8 的推論結果。
2. 計算兩者的 Top-1 準確率差異，並分析 INT8 量化的適用情況。
