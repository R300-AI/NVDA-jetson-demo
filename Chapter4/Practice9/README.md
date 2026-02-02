# **比較原生模型與 TorchScript 模型的效能差異**

# 題目說明
在 **NVIDIA Jetson Orin** 上比較：

| 模式 | 特性 | 教材說明 |
| --- | --- | --- |
| **Eager（原生 PyTorch）** | 動態計算、適合開發，但含 Python overhead | 官方指出 Eager 因 Python interpreter 會有額外負擔 |
| **TorchScript — trace** | 以範例輸入記錄固定運算路徑；**速度通常比 script 快一些** | trace 適用於固定架構，如 SmallCNN；教材示例指出 trace 稍快於 script  
| **TorchScript — script** | 直接編譯 Python 程式碼；保留控制流（if/loop） | script 可處理動態控制流；trace 不安全時必須用 script |

測試重點：

- **公平比較前需完成 GPU 預熱**
- **在 CUDA 上量測效能需使用 `torch.cuda.synchronize()`**，避免 GPU 非同步造成量測誤差 
- 最終比較三種模式的 **samples/sec**，體驗 TorchScript 的加速與可部署性優勢

---

# 作業練習

### **① 實測三種推論效能**
請紀錄：
- Eager samples/sec
- Traced samples/sec
- Scripted samples/sec

並回答：
- **哪個最快？差異多少？**
- trace 是否如講義所說稍微快於 script？（固定結構下通常如此） 

---

### **② 使用 tegrastats 觀察 GPU 活動**
程式執行時開啟： `sudo tegrastats` ，並記錄：
- GPU 使用率
- GPU 時脈
- GPU memory 使用量

反思：TorchScript 是否使 GPU 更穩定飽和？

---

### **③ 嘗試不同 batch size（32, 64, 128）**
觀察：
- 大 batch 時 trace/script 加速幅度是否更明顯？
- 是否如講義所說，固定結構 CNN 更適合 trace？  

---

### **⑤ 嘗試儲存／載入並跨程式使用 TorchScript**
可獨立部署模型，擺脫 Python 環境
