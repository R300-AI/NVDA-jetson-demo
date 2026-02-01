"""
Practice 5: INT8 量化精度驗證

前置條件:
    需要 Practice 2 的 yolov8n_fp32.engine
    需要 Practice 4 的 yolov8n_int8.engine

執行方式:
    python3 validate_int8.py
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# ========== 設定參數 ==========
fp32_engine_path = "yolov8n_fp32.engine"
int8_engine_path = "yolov8n_int8.engine"
input_shape = (1, 3, 640, 640)         # YOLOv8n 輸入形狀

print("【Practice 5】比較 FP32 與 INT8 推論精度")


# ========== TODO 1: 載入兩個引擎 ==========
# 提示: 參考 Practice 3 的載入方式

logger = None                          # 請建立 Logger
fp32_engine = None                     # 請載入 FP32 引擎
int8_engine = None                     # 請載入 INT8 引擎


# ========== TODO 2: 執行推論並比較 ==========
# 提示: 
# 1. 產生相同的測試輸入 (np.random.rand)
# 2. 分別用 FP32 和 INT8 引擎推論
# 3. 計算輸出差異

test_input = np.random.rand(*input_shape).astype(np.float32)

fp32_output = None                     # 請完成 FP32 推論
int8_output = None                     # 請完成 INT8 推論


# ========== 計算差異 ==========
# mse = np.mean((fp32_output - int8_output) ** 2)
# max_abs_diff = np.max(np.abs(fp32_output - int8_output))
# print(f"MSE (均方誤差): {mse:.6f}")
# print(f"最大絕對誤差: {max_abs_diff:.6f}")
