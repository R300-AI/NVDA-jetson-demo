"""
Practice 4: INT8 量化校正 (Post-Training Quantization, PTQ)

題目說明:
1. 使用 Practice 2 產生的 yolov8n.onnx 模型
2. 建立 data_loader.py 腳本，產生隨機數據作為校正資料
3. 使用 Polygraphy 進行 INT8 校正，產生 calibration cache
4. 使用 trtexec 載入 cache 編譯 INT8 引擎

執行方式:
    # Step 1: 使用 Polygraphy 產生 calibration cache
    polygraphy convert yolov8n.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache yolov8n_calib.cache

    # Step 2: 使用 trtexec 編譯 INT8 引擎
    trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \
            --saveEngine=yolov8n_int8.engine

    # Step 3: 效能分析
    trtexec --loadEngine=yolov8n_int8.engine --dumpProfile
"""

import numpy as np


def load_data():
    """
    Polygraphy data loader function.
    產生隨機數據作為校正資料（示範用途）。
    
    注意：實際應用中應使用真實的代表性資料進行校正，
    以獲得更好的量化效果。
    """
    num_samples = 100  # 校正樣本數量
    
    for _ in range(num_samples):
        # 產生隨機輸入（YOLOv8 輸入為 640x640 RGB 圖像）
        img_array = np.random.rand(1, 3, 640, 640).astype(np.float32)
        
        # YOLOv8 的輸入名稱為 "images"
        yield {"images": img_array}


if __name__ == "__main__":
    print("=" * 60)
    print("Practice 4: INT8 量化校正 (PTQ) with Polygraphy")
    print("=" * 60)

    print("\n此腳本定義了 load_data() 函數供 Polygraphy 使用")
    print("\n使用步驟:")
    print("1. 使用 Polygraphy 產生 calibration cache:")
    print("   polygraphy convert yolov8n.onnx --int8 \\")
    print("       --data-loader-script ./data_loader.py \\")
    print("       --calibration-cache yolov8n_calib.cache")
    print("\n2. 使用 trtexec 編譯 INT8 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \\")
    print("           --saveEngine=yolov8n_int8.engine")
    print("\n3. 效能分析:")
    print("   trtexec --loadEngine=yolov8n_int8.engine --dumpProfile")
    print("\n4. 比較 FP16 與 INT8 (PTQ) 的效能差異")
    print("\n注意: 本範例使用隨機數據，實際應用中應使用真實資料進行校正。")
