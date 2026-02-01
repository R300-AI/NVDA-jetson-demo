# ResNet18 的 QAT with CIFAR-10

### 題目說明
1. 使用 torchvision.datasets.CIFAR10 載入 CIFAR-10 訓練集與測試集，並進行標準化處理
2. 建立 ResNet18 模型，修改最後一層 fc 輸出大小為 10。
3. 在模型中插入 QuantStub 與 DeQuantStub，並設定 qconfig，執行 prepare_qat。
4. 在 CIFAR-10 訓練集上進行 QAT 訓練，模擬量化誤差。
5. 完成 QAT 訓練後，將模型轉換為量化版本並匯出成 ONNX 格式 (opset_version=13)。
6. 使用 trtexec 工具將 ONNX 模型編譯成 INT8 精度的 TensorRT 引擎，並加上 --dumpLayerInfo 與 --dumpProfile，輸出每一層的部署資訊與效能

### 作業練習
比較 ResNet18 的 FP32、PTQ、QAT 三種版本在 CIFAR-10 測試集上的推論效能與準確率，並分析量化對效能與精度的影響。
