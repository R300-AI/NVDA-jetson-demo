"""
Lab 2: 匯出 YOLOv8 模型（Google Colab 用）

⚠️ 請在 Google Colab 執行此程式碼

使用方式:
1. 開啟 Google Colab: https://colab.research.google.com/
2. 將此檔案內容貼到 Colab 並執行
3. 下載 yolov8n.onnx，傳輸到 Jetson

在 Jetson 上編譯:
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16

執行推論與效能分析:
    trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile --exportProfile=yolov8n_fp32_profile.json
    trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile --exportProfile=yolov8n_fp16_profile.json
"""

# 安裝 ultralytics（在 Colab 中取消註解執行）
# !pip install ultralytics

from ultralytics import YOLO

# 載入並匯出模型
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=17, imgsz=640, simplify=True)

print("模型已匯出為 yolov8n.onnx")

# 下載檔案（Colab 專用，取消註解執行）
# from google.colab import files
# files.download('yolov8n.onnx')
