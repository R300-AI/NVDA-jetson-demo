建立CNN模型做正反傳播
---

**題目說明**
1. 建立簡化版 **SmallCNN**：`Conv → ReLU → MaxPool → Conv → ReLU → Flatten → FC`，輸入為 `N×3×32×32`。
2. 以**合成資料**建立一個 batch，完成 **Forward → Loss → Backward → Optimizer Step** 的完整鏈。
3. 量測並印出各階段的關鍵結果（logits 形狀、loss、梯度示例、更新後 loss），在 **GPU 可用時使用 CUDA** 執行並遵循 **同步量測**的良好習慣（如需要精準量測時，於 CUDA 端加入同步以確保統計正確）。。


**作業練習**
&#10004;成功建立CNN模型
&#10004;完成前向傳播
&#10004;完成反向傳播[梯度計算]
&#10004;執行Optimizer更新
