"""
Practice 4: INT8 量化校正 (Post-Training Quantization, PTQ)

題目說明:
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset_version=17)
2. 建立 data_loader.py 腳本，產生隨機校正資料
3. 使用 Polygraphy 進行 INT8 編譯，產生 calibration cache
4. 使用 trtexec --dumpLayerInfo 與 --dumpProfile 輸出效能數據

執行方式:
    # Step 1: 匯出 ONNX
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=17, imgsz=640)"

    # Step 2: 使用 Polygraphy 編譯 INT8 引擎
    polygraphy convert yolov8n.onnx --int8 \\
        --data-loader-script ./data_loader.py \\
        --calibration-cache yolov8n_calib.cache \\
        -o yolov8n_int8.engine

    # Step 3: 效能分析
    trtexec --loadEngine=yolov8n_int8.engine --dumpProfile --dumpLayerInfo
"""

import numpy as np

def load_data():
    """
    Polygraphy data loader function.
    產生校正資料供 INT8 量化使用。
    """
    # ========== TODO: 實作校正資料產生器 ==========
    # 產生 100 個批次的隨機校正資料
    # 資料形狀: (1, 3, 640, 640)，資料型態: float32，範圍: [0.0, 1.0]
    # 注意: YOLOv8 的輸入名稱為 "images"
    """
    提示:
        for _ in range(100):
            yield {"images": np.random.rand(1, 3, 640, 640).astype(np.float32)}
    """
    pass  # 請修改此行


if __name__ == "__main__":
    print("=" * 60)
    print("Practice 4: INT8 量化校正 (PTQ) with Polygraphy")
    print("=" * 60)

    print("\n此腳本定義了 load_data() 函數供 Polygraphy 使用")
    print("\n使用步驟:")
    print("1. 匯出 ONNX 模型:")
    print('   python3 -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\').export(format=\'onnx\', opset=17, imgsz=640)"')
    print("\n2. 使用 Polygraphy 編譯 INT8 引擎:")
    print("   polygraphy convert yolov8n.onnx --int8 \\")
    print("       --data-loader-script ./data_loader.py \\")
    print("       --calibration-cache yolov8n_calib.cache \\")
    print("       -o yolov8n_int8.engine")
    print("\n3. 效能分析:")
    print("   trtexec --loadEngine=yolov8n_int8.engine --dumpProfile --dumpLayerInfo")
    print("\n4. 比較 FP16 與 INT8 (PTQ) 的效能差異")
