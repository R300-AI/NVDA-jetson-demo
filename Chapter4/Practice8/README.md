# **測試 JIT 編譯後的推論效能**

# 題目說明
在 **NVIDIA Jetson Orin** 上，使用 PyTorch 的 **JIT 編譯技術（trace & script）**，比較下列推論速度（samples/sec）：
- **原生 Eager 模式**
- **trace 編譯後模型**
- **script 編譯後模型**

### 重點概念
- **trace**：以「範例輸入」記錄一次運算路徑，適用固定結構 CNN。
- **script**：解析 Python 原始碼，能保留 `if/for` 等控制流。
- **CUDA 同步** (`torch.cuda.synchronize()` )：避免 GPU 非同步執行造成量測誤差，是 Jetson 上效能測試必做步驟。

---

# 作業練習

### **1. 測試三種推論效能**
請實測以下三種模式的吞吐量（samples/sec），並比較差異：
- Eager（未 JIT）
- JIT trace
- JIT script

---

### **2. 使用 tegrastats 觀察 GPU 行為**
請在 Jetson 上同時執行： `sudo tegrastats` 觀察：
- GPU 使用率
- GPU 時脈
- Memory 使用量

並記錄 JIT 前後是否有差異。

---

### **3. 嘗試不同 batch size：32 / 64 / 128**
觀察 batch 增大時：
- Eager / Trace / Script 的速度是否差異更加明顯？
- GPU 是否更容易被吃滿？
