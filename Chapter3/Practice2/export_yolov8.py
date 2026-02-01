"""
Practice 2: 匯出 YOLOv8 模型並比較 FP32 與 FP16 效能

========================================
⚠️ 請在 Google Colab 執行此程式碼 ⚠️
========================================

由於 NVIDIA 官方 PyTorch wheel 未包含 torchvision，
而 ultralytics 套件依賴 torchvision，因此無法在 Jetson 上直接執行。

使用方式:
1. 開啟 Google Colab: https://colab.research.google.com/
2. 將此檔案內容貼到 Colab 並執行
3. 下載產生的 yolov8n.onnx 檔案
4. 傳輸到 Jetson 進行 TensorRT 編譯

在 Jetson 上編譯 TensorRT 引擎:
    # FP32 精度
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine
    
    # FP16 精度
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16

執行推論與效能分析:
    trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile --exportProfile=yolov8n_fp32_profile.json
    trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile --exportProfile=yolov8n_fp16_profile.json
"""

# ===== 以下程式碼請在 Google Colab 執行 =====

# Step 1: 安裝 ultralytics（在 Colab 中執行）
# !pip install ultralytics

from ultralytics import YOLO

def main():
    print("=" * 60)
    print("Practice 2: 匯出 YOLOv8n 為 ONNX 格式")
    print("=" * 60)

    # ========== TODO 1: 載入 YOLOv8n 預訓練模型 ==========
    # 提示: model = YOLO('yolov8n.pt')
    model = None  # 請修改此行


    # ========== TODO 2: 匯出 ONNX 模型 ==========
    """
    請使用 model.export 匯出模型
    提示:
        model.export(
            format='onnx',
            opset=17,
            imgsz=640,
            simplify=True
        )
    """


    print("\n模型已匯出為 yolov8n.onnx")
    
    # Step 3: 下載 ONNX 檔案（Colab 專用）
    # from google.colab import files
    # files.download('yolov8n.onnx')
    
    print("\n下一步（在 Jetson 上執行）:")
    print("1. 將 yolov8n.onnx 傳輸到 Jetson")
    print("2. 編譯 FP32 TensorRT 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine")
    print("\n3. 編譯 FP16 TensorRT 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16")
    print("\n4. 執行推論與效能分析:")
    print("   trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile")
    print("   trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile")

if __name__ == "__main__":
    main()
