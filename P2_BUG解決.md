1️⃣ 檢查安裝位置

執行：

pip3 show ultralytics



3️⃣ 永久生效（可選）

如果每次開機都想用 yolo，可以把上面 export 加到 ~/.bashrc：

echo 'export PATH=$PATH:/home/user11/.local/bin' >> ~/.bashrc
source ~/.bashrc

(以下指令照搬 bash )

#!/bin/bash
export PATH=/usr/src/tensorrt/bin:$PATH

echo "=== 步驟 1: 匯出 YOLOv8n ONNX 模型 ==="
yolo export model=yolov8n.pt format=onnx opset=13

echo "=== 步驟 2: 編譯 FP32 Engine (基準線) ==="
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_fp32.plan \
        --dumpProfile > yolo_fp32.log

echo "=== 步驟 3: 編譯 FP16 Engine (加速版) ==="
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_fp16.plan \
        --fp16 \
        --dumpProfile > yolo_fp16.log

echo "執行完成！請比較 fp32 與 fp16 的 log 檔案。"

