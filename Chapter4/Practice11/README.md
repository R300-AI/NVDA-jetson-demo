# **效能 Benchmark：原生 vs TensorRT**

#  題目說明
### ● **原生 PyTorch Eager 模型（CPU/GPU）**

- 以 Python 動態執行，存在 Python overhead
- 作為基準線效能

### ● **TensorRT Engine（FP16）推論效能**

教材指出 TensorRT 可透過：

- Layer/Tensor Fusion
- Kernel Auto-Tuning
- FP16/INT8 精度加速
- Dynamic Tensor Memory
達成更高的吞吐量與更低延遲，是 **Jetson Orin 邊緣部署的最佳化路徑**。 

實作流程：

1. 以 SmallCNN 建立 **Eager 基準線**（PyTorch 手動推論）
2. 將模型 **匯出 ONNX**（使用 `torch.onnx.export`）
3. 使用 **trtexec** 生成 **FP16 TensorRT engine (.plan)**
4. 使用 trtexec 再量測 engine 效能（Throughput / percentile latency）
5. 比較兩者差異

---

# 作業練習
---

### **① 記錄兩種模式的 samples/sec**

- PyTorch Eager
- TensorRT FP16 engine（使用 trtexec 測試）

比較吞吐量差異：

> TensorRT 是否比 Eager 明顯更快？差幾倍？

---

### **② 使用 tegrastats 觀察推論期間 GPU 行為**

在程式執行期間執行： `sudo tegrastats`，並觀察：

- GPU 使用率
- GPU 時脈
- Memory 使用量
- TensorRT engine 推論時是否較為穩定或保持高頻？

---

### **③ 修改 batch size（如 32 / 64 / 128）**

重新執行 `trtexec --loadEngine`，觀察：

- 哪個 batch size 給 TensorRT 最高 Throughput（samples/sec）？
- 是否如講義所述：**大 batch 更能餵飽 TensorRT engine**？
