檢查 GPU 記憶體使用狀況
---

## 題目說明 ##

目標在 **NVIDIA Jetson Orin** 上利用 **PyTorch 的 CUDA 記憶體 API**，搭配簡易 CNN 模型，觀察：

### ● GPU 記憶體分配（allocated）
PyTorch **實際用於 tensor/模型參數/activation 的顯存**。

### ● GPU 記憶體保留（reserved）
PyTorch CUDA allocator **向系統預先申請但尚未使用的顯存**，可能造成 OOM 誤解。

### ● Forward / Backward / Optimizer 各階段顯存變化
教材中強調這些階段各會累積 activation、gradient、momentum 等記憶體。 

### ● empty_cache() 的效果
釋放未使用的 GPU cache，但不會釋放真正被張量占用的 allocated memory


## 作業練習
### **① 記錄五個階段的 allocated / reserved (MB)**
- 初始
- Forward 後
- Backward 後
- Optimizer Step 後
- empty_cache() 後

hint: reserved 可能大於 allocated，因此要清楚區分兩者。 


### **② 利用 tegrastats 同步觀察**

執行: 'sudo tegrastats'，並記錄：
- GPU memory 使用率
- GPU 頻率
- CPU 使用率
- module 對 GPU 記憶體造成的尖峰


### **③ 嘗試不同 batch size（例如 32 / 64 / 128）**
- activation 成長與 batch size 成正比
- 大 batch size 更可能 OOM


### **④ 移除 Backward（推論模式）比較差異**
把： 'loss.backward()' 註解掉，觀察 reserved 與 allocated 減少多少。
hint: backward 階段會產生大量 gradient 記憶體。
