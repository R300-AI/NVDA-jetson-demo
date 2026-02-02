測試不同Batch size對效能的影響
---

**題目說明**
目標是比較 **不同 batch size** 在 **Jetson Orin GPU** 上的吞吐量（samples/sec），理解 GPU 在不同批量下的效能表現。
1. 建立一個 SmallCNN
2. 使用不同 batch size（16 / 32 / 64 / 128 / 256 / 512）產生合成資料
3. 在 GPU 上執行固定 iteration（200 次）
4. 量測每種 batch size 的 **吞吐量（samples/sec）**
5. 輸出比對結果

**作業練習**
&#10004;在Jetson Orin 上實現 GPU 效能量測 
&#10004;掌握GPU吞吐量計算公式: Batch size增加 →吞吐量通常上升，但記憶體需求也會增加
&#10004;最佳化建議: 可依 GPU 記憶體大小調整最大 Batch Size
&#10004;能避免 GPU OOM 與 memory bottleneck 

