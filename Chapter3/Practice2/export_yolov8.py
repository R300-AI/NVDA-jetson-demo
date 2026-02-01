"""
Practice 2: 匯出 YOLOv8 模型並比較 FP32 與 FP16 效能

⚠️ 請在 Google Colab 執行
"""

# !pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# TODO: 匯出 ONNX

# from google.colab import files
# files.download('yolov8n.onnx')
