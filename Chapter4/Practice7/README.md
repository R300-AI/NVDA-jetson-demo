# **將模型轉換為 TorchScript 格式**

## 題目說明

- 了解 **TorchScript 兩種轉換方法**：`torch.jit.trace`（結構固定、以範例輸入錄製路徑）與 `torch.jit.script`（解析原始 Python，**保留 if/for 等控制流**）。 
- 在 **NVIDIA Jetson Orin** 上完成 **Eager vs. Traced vs. Scripted** 的 **推論吞吐量（samples/sec）** 粗測，並學會將模型 **儲存為 `.ts`、載入回測** 的完整流程。 
- 知道何時用 `trace`、何時必須用 `script`（例如含 **資料相依控制流** 的 `DynamicHead`），並在 GPU 上進行正確的 **CUDA 同步** 以確保量測時間可信。

## 作業練習

1. **正確選擇 trace/script：**
    - 以預設 `SmallCNN`（結構固定）執行 `trace` 與 `script`，比較二者的 samples/sec。
    - 啟用 `-use_dynamic`（使用 `DynamicHead`），觀察：為何 **trace** 會被程式刻意略過？（因為包含資料相依分支，應使用 **script**。） 
2. **效能量測與 CUDA 同步：**
    - 以 `bench_infer()` 的 **預熱 + 多次迭代 +（必要時）CUDA 同步** 方法比對 Eager / Traced / Scripted 速度差異。
    - 說明為何 GPU 量測前後需要 `torch.cuda.synchronize()` 以避免非同步造成的時間誤差。 
3. **模型序列化與回載：**
    - 使用 `torch.jit.save` 將兩種 TorchScript 版本輸出為 `.ts`，再以 `torch.jit.load` 回載並重測吞吐量，確認部署可重現且穩定。
