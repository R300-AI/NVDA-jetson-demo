"""
Practice 4: INT8 量化校正 Data Loader

執行方式:
    polygraphy convert yolov8n.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache yolov8n_calib.cache
"""

import numpy as np


def load_data():
    """產生校正資料批次"""
    num_samples = 100
    
    # TODO: 產生隨機校正資料並 yield
    pass

