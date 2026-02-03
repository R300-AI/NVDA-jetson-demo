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
