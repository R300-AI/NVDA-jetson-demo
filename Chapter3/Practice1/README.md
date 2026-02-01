# TensorRT基本部署流程 (to GPU)

### 題目說明
1. 使用純 PyTorch 自訂 CNN 分類器並匯出為 ONNX 格式 (opset_version=17)。
2. 使用 trtexec 將 ONNX 模型檔編譯成 FP32 的 TensorRT 引擎指令 (--shapes=input:1x3x224x224)。
3. 使用 trtexec 工具執行推論，並加上 --dumpProfile，列出每一層的執行時間與資源使用情況。

### 作業練習
解析 profile 輸出，觀察各層的執行時間分布，找出耗時最多的層並分析其原因。
