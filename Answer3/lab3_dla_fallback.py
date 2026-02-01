"""
Practice 3: DLA Operator 驗證與 Fallback 練習

題目說明:
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset_version=13)
2. 使用 trtexec 工具將 ONNX 模型編譯成 INT8 精度的 TensorRT 引擎
3. 使用 --dumpLayerInfo 輸出每層的部署資訊，模擬 DLA 部署流程

注意: Jetson Orin Nano 未搭載 DLA，本練習以 GPU 模擬 DLA 的操作流程，
      讓學生了解 DLA 支援的運算子限制與 Fallback 機制。

執行方式:
    python3 dla_fallback.py

編譯 TensorRT 引擎（模擬 DLA 部署）:
    # 在有 DLA 的裝置上會使用: --useDLACore=0 --allowGPUFallback
    # Orin Nano 無 DLA，改用 GPU 並觀察層資訊
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_int8_gpu.engine \\
            --int8 --shapes=images:1x3x640x640 \\
            --dumpLayerInfo --exportLayerInfo=yolov8n_layers.json

DLA 支援的運算子（參考）:
    - Convolution (Conv)
    - Deconvolution (ConvTranspose)
    - Fully Connected (Gemm)
    - Pooling (MaxPool, AveragePool)
    - Activation (ReLU, Sigmoid, Tanh)
    - BatchNormalization
    - Scale
    - ElementWise (Add, Sub, Mul, Max, Min)
    - Concatenation

DLA 不支援的運算子（會 Fallback 到 GPU）:
    - Softmax
    - Resize (Upsample)
    - Split
    - Pad (某些模式)
    - ReduceMean, ReduceMax 等
"""

from ultralytics import YOLO

def main():
    print("=" * 60)
    print("Practice 3: DLA Operator 驗證與 Fallback 練習")
    print("=" * 60)

    print("\n注意: Jetson Orin Nano 未搭載 DLA")
    print("本練習以 GPU 模擬 DLA 的操作流程，觀察層級部署資訊")

    # ========== TODO 1: 載入 YOLOv8n 預訓練模型 ==========
    # 提示: model = YOLO('yolov8n.pt')
    model = YOLO('yolov8n.pt')


    # ========== TODO 2: 匯出 ONNX 模型 ==========
    """
    請使用 model.export 匯出模型
    提示:
        model.export(
            format='onnx',
            opset=13,
            imgsz=640,
            simplify=True
        )
    """
    model.export(
        format='onnx',
        opset=13,
        imgsz=640,
        simplify=True
    )


    print("\n模型已匯出")
    print("\n下一步:")
    print("1. 編譯 INT8 TensorRT 引擎並輸出層資訊:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_int8.engine \\")
    print("           --int8 --shapes=images:1x3x640x640 \\")
    print("           --dumpLayerInfo --exportLayerInfo=yolov8n_layers.json")
    print("\n2. 使用 Netron 開啟 yolov8n.onnx 檢視模型結構")
    print("\n3. 分析 yolov8n_layers.json，找出哪些運算子不支援 DLA")
    print("\nDLA 不支援的常見運算子:")
    print("   - Softmax")
    print("   - Resize / Upsample")
    print("   - Split")
    print("   - Certain Pad modes")

if __name__ == "__main__":
    main()
