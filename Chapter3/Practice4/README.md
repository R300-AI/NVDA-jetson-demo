# 透過 Polygraphy 實現 INT8 PTQ

### 題目說明
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset_version=17)。
2. 建立 `data_loader.py` 腳本，使用 numpy 產生大小為 [1, 3, 640, 640] 的隨機校正資料。
3. 使用 Polygraphy 工具進行 INT8 編譯，並產生 calibration cache 檔案。
4. 使用 trtexec 工具加上 --dumpLayerInfo 與 --dumpProfile，輸出每一層的部署資訊與效能數據。

### 作業練習
* 觀察量化後的推論效能與準確率，並比較 FP16 與 INT8 (PTQ) 的差異，分析量化對效能與精度的影響。
