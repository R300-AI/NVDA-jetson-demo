"""
Practice 4: INT8 量化校正 Data Loader

前置條件:
    需要 Practice 2 產生的 yolov8n.onnx

執行方式:
    polygraphy convert yolov8n.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache yolov8n_calib.cache \
        -o yolov8n_int8.engine

    （產生 cache 後，也可用 trtexec 編譯）
    trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache --saveEngine=yolov8n_int8.engine
    trtexec --loadEngine=yolov8n_int8.engine --dumpProfile --dumpLayerInfo
"""

import numpy as np


def load_data():
    """產生校正資料批次
    
    YOLOv8n 模型資訊:
        輸入名稱: "images"（可用 Netron 開啟 ONNX 查看）
        輸入形狀: (1, 3, 640, 640)
    """
    # ========== TODO: 產生隨機校正資料 ==========
    # 提示: 使用 yield 產生字典，格式為 {"輸入名稱": np.array}
    # 1. key 名稱須與 ONNX 輸入名稱一致（YOLOv8 為 "images"）
    # 2. 形狀須與模型輸入一致: (1, 3, 640, 640)
    # 3. 資料型態須為 np.float32
    # 4. 建議產生 100~500 個樣本
    
    num_samples = 100                  # 校正樣本數量
    input_name = "images"              # YOLOv8 輸入名稱
    input_shape = (1, 3, 640, 640)     # YOLOv8 輸入形狀
    
    for i in range(num_samples):
        data = np.random.rand(*input_shape).astype(np.float32)
        yield {input_name: data}
