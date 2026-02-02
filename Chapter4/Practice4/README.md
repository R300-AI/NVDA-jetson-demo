比較CPU與GPU的訓練速度
---

**題目說明**
-`model.to(device)`、`tensor.to(device)` 啟用 GPU 加速
-自動混合精度 **AMP (torch.cuda.amp)** 進一步提升速度
-`torch.cuda.synchronize()` 讓 CUDA 量測時間準確
- 不同批次大小會影響資料載入與 GPU 飽和度

**訓練流程：**
1. 以 SmallCNN 建立簡單分類模型
2. 產生合成資料（64×3×32×32）
3. **CPU 執行單 epoch（100 iteration）**
4. **GPU 執行單 epoch（AMP 啟用可加速）**
5. 比較 CPU / GPU 訓練時間與加速比
