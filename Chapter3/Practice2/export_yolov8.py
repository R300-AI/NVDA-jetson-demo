"""
Practice 2: 匯出 YOLOv8 模型並比較 FP32 與 FP16 效能

題目說明:
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset=17)
2. 使用 trtexec 將 ONNX 模型編譯成 FP32 及 FP16 兩個不同精度的 TensorRT 引擎
3. 使用 trtexec 執行推論並加上 --dumpProfile 分析效能

執行方式:
    python3 export_yolov8.py

編譯 TensorRT 引擎:
    # FP32 精度
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine
    
    # FP16 精度
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16

執行推論與效能分析:
    trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile --exportProfile=yolov8n_fp32_profile.json
    trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile --exportProfile=yolov8n_fp16_profile.json
"""

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


    print("\n模型已匯出")
    print("\n下一步:")
    print("1. 編譯 FP32 TensorRT 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp32.engine")
    print("\n2. 編譯 FP16 TensorRT 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16")
    print("\n3. 執行推論與效能分析:")
    print("   trtexec --loadEngine=yolov8n_fp32.engine --dumpProfile")
    print("   trtexec --loadEngine=yolov8n_fp16.engine --dumpProfile")

if __name__ == "__main__":
    main()
