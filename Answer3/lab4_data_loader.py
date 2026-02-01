"""
Lab 4: INT8 量化校正 Data Loader

執行方式:
    polygraphy convert yolov8n.onnx --int8 \
        --data-loader-script ./lab4_data_loader.py \
        --calibration-cache yolov8n_calib.cache

    trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \
            --saveEngine=yolov8n_int8.engine
"""

import numpy as np


def load_data():
    """Polygraphy data loader - 產生隨機校正資料"""
    num_samples = 100
    
    for _ in range(num_samples):
        img_array = np.random.rand(1, 3, 640, 640).astype(np.float32)
        yield {"images": img_array}  # YOLOv8 輸入名稱
