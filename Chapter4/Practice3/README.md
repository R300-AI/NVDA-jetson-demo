以PyTorch實現動態計算(dynamic computatuon ) 以及靜態計算(static computaiton)
---
**題目說明**
1. **動態計算（Eager）**：以**資料相依**的 `if/for` 控制流（由張量值決定分支與迴圈次數）驅動運算；易於研究與除錯、撰寫彈性高。 
2. **靜態計算（TorchScript）**：將模型**凍結為可序列化／可優化的圖**，便於部署與跨語言推論；
    - `torch.jit.script`：**保留控制流**（可處理 data‑dependent `if/loop`）。
    - `torch.jit.trace`：以**範例輸入**記錄**一次**運算路徑，**不保留**資料相依控制流，適合結構固定模型。 
3. **Jetson Orin 實務提醒**：做推論效能或延遲比較時，因 CUDA **預設非同步**，需適時 `torch.cuda.synchronize()` 以取得準確量測。

**作業練習**
&#10004;完成動態計算
&#10004;完成靜態計算

