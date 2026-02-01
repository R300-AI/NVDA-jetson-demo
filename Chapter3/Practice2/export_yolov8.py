"""
Practice 2: 匯出 YOLOv8 模型並比較 FP32 與 FP16 效能

⚠️ 請在 Google Colab 執行此腳本（Jetson 無 torchvision）

執行方式 (Colab):
    1. 上傳此檔案到 Colab 或複製程式碼執行
    2. 下載產生的 yolov8n.onnx 到 Jetson

執行方式 (Jetson):
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
    trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile --dumpLayerInfo
    trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile --dumpLayerInfo
"""

# ========== 安裝套件（Colab 環境）==========
# !pip install ultralytics

from ultralytics import YOLO

print("【Practice 2】匯出 YOLOv8n 為 ONNX 格式")

# ========== TODO: 匯出 ONNX ==========
# 提示: 使用 model.export() 方法
# 參數: format='onnx', opset=17, imgsz=640, simplify=True

model = YOLO("yolov8n.pt")
model.export()                         # 請填入正確的匯出參數

# ========== 下載檔案（Colab 環境）==========
# from google.colab import files
# files.download('yolov8n.onnx')

print("已匯出: yolov8n.onnx")
print("輸入名稱: images, 形狀: (1, 3, 640, 640)")
print("請下載 yolov8n.onnx 到 Jetson，接著執行:")
print("  trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine")
print("  trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16")
