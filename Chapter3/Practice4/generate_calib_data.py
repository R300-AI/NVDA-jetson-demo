"""
Practice 4: INT8 量化校正 (Post-Training Quantization, PTQ)

題目說明:
1. 使用 Ultralytics 套件匯出 yolov8n.pt 模型為 ONNX 格式 (opset_version=13)
2. 建立隨機校正資料 (1, 3, 640, 640)，並儲存為二進位檔案 calib.bin
3. 使用 trtexec 進行 INT8 編譯，讓 TensorRT 根據 calib.bin 生成 calib.cache
4. 使用 --dumpLayerInfo 與 --dumpProfile 輸出效能數據

執行方式:
    python3 generate_calib_data.py

編譯 TensorRT 引擎:
    trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_int8_ptq.engine \\
            --int8 --shapes=images:1x3x640x640 \\
            --calib=calib.cache \\
            --dumpProfile --dumpLayerInfo
"""

import numpy as np

def main():
    print("=" * 60)
    print("Practice 4: INT8 量化校正 (PTQ)")
    print("=" * 60)

    # ========== TODO 1: 建立隨機校正資料 ==========
    # 資料形狀: (1, 3, 640, 640)，資料型態: float32，範圍: [0.0, 1.0]
    """
    請建立隨機校正資料
    提示:
        calib_data = np.random.uniform(0.0, 1.0, (1, 3, 640, 640)).astype(np.float32)
    """
    calib_data = None  # 請修改此行


    # ========== TODO 2: 將校正資料儲存為二進位檔案 ==========
    calib_path = "calib.bin"
    """
    請將校正資料儲存為二進位檔案
    提示:
        calib_data.tofile(calib_path)
    """


    print(f"\n校正資料已儲存至: {calib_path}")
    print(f"資料形狀: {calib_data.shape if calib_data is not None else 'N/A'}")
    print(f"資料型態: {calib_data.dtype if calib_data is not None else 'N/A'}")

    print("\n下一步:")
    print("1. 先匯出 ONNX 模型 (若尚未匯出):")
    print("   python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=13, imgsz=640)\"")
    print("\n2. 編譯 INT8 TensorRT 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_int8_ptq.engine \\")
    print("           --int8 --shapes=images:1x3x640x640 \\")
    print("           --calib=calib.cache \\")
    print("           --dumpProfile --dumpLayerInfo")
    print("\n3. 比較 FP16 與 INT8 (PTQ) 的效能差異")

if __name__ == "__main__":
    main()
