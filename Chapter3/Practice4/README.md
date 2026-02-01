# 透過PyTorch實現一個基本的QAT

### 題目說明
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset_version=13)。
2. 利用 std::vector<float> 及 std::generate 建立大小為 [1, 3, 640, 640] 的隨機向量(資料型態為float32，範圍 [0.0, 1.0])，並將結果輸出成二進位檔案 calib.bin。
3. 使用 trtexec 工具進行 INT8 編譯，指定校正資料檔案路徑讓 TensorRT 根據 calib.bin 生成 calib.cache 。
4. 使用 trtexec 工具加上 --dumpLayerInfo 與 --dumpProfile，輸出每一層的部署資訊與效能數據。
### 作業練習
* 觀察量化後的推論效能與準確率，並比較 FP16 與 INT8 (PTQ) 的差異，分析量化對效能與精度的影響。
