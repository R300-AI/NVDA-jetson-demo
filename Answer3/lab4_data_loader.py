"""
Practice 4: INT8 量化校正 (Post-Training Quantization, PTQ)

題目說明:
1. 使用 Practice 2 產生的 yolov8n.onnx 模型
2. 建立 data_loader.py 腳本，從 calib_images/ 載入真實校正圖片
3. 使用 Polygraphy 進行 INT8 校正，產生 calibration cache
4. 使用 trtexec 載入 cache 編譯 INT8 引擎

執行方式:
    # Step 1: 準備校正圖片（放入 calib_images/ 資料夾，建議 100-500 張）
    mkdir calib_images
    # 從 COCO 或 ImageNet 下載代表性圖片

    # Step 2: 使用 Polygraphy 產生 calibration cache
    polygraphy convert yolov8n.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache yolov8n_calib.cache

    # Step 3: 使用 trtexec 編譯 INT8 引擎
    trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \
            --saveEngine=yolov8n_int8.engine

    # Step 4: 效能分析
    trtexec --loadEngine=yolov8n_int8.engine --dumpProfile
"""

import numpy as np
from PIL import Image
import os


def load_data():
    """
    Polygraphy data loader function.
    從 calib_images/ 資料夾載入真實圖片作為校正資料。
    """
    calib_dir = "./calib_images"
    
    # 取得所有圖片檔案
    img_files = [f for f in os.listdir(calib_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for img_name in img_files[:100]:
        img_path = os.path.join(calib_dir, img_name)
        
        # 載入並調整大小（YOLOv8 輸入為 640x640）
        img = Image.open(img_path).convert('RGB').resize((640, 640))
        
        # 轉換為 numpy array 並正規化
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 加入 batch 維度
        img_array = np.expand_dims(img_array, axis=0)
        
        # YOLOv8 的輸入名稱為 "images"
        yield {"images": img_array}


if __name__ == "__main__":
    print("=" * 60)
    print("Practice 4: INT8 量化校正 (PTQ) with Polygraphy")
    print("=" * 60)

    print("\n此腳本定義了 load_data() 函數供 Polygraphy 使用")
    print("\n使用步驟:")
    print("1. 準備校正圖片（放入 calib_images/ 資料夾）:")
    print("   mkdir calib_images")
    print("   # 從 COCO 或 ImageNet 下載 100-500 張代表性圖片")
    print("\n2. 使用 Polygraphy 產生 calibration cache:")
    print("   polygraphy convert yolov8n.onnx --int8 \\")
    print("       --data-loader-script ./data_loader.py \\")
    print("       --calibration-cache yolov8n_calib.cache")
    print("\n3. 使用 trtexec 編譯 INT8 引擎:")
    print("   trtexec --onnx=yolov8n.onnx --int8 --calib=yolov8n_calib.cache \\")
    print("           --saveEngine=yolov8n_int8.engine")
    print("\n4. 效能分析:")
    print("   trtexec --loadEngine=yolov8n_int8.engine --dumpProfile")
    print("\n5. 比較 FP16 與 INT8 (PTQ) 的效能差異")
