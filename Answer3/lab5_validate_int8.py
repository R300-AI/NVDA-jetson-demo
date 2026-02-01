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

logger = trt.Logger(trt.Logger.WARNING)
with open(fp32_engine_path, "rb") as f:
    fp32_engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
with open(int8_engine_path, "rb") as f:
    int8_engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())


# ========== TODO 2: 執行推論並比較 ==========
# 提示: 
# 1. 產生相同的測試輸入 (np.random.rand)
# 2. 分別用 FP32 和 INT8 引擎推論
# 3. 計算輸出差異

test_input = np.random.rand(*input_shape).astype(np.float32)
output_shape = (1, 84, 8400)           # YOLOv8n 輸出形狀

# FP32 推論
fp32_context = fp32_engine.create_execution_context()
fp32_output = np.empty(output_shape, dtype=np.float32)
d_input = cuda.mem_alloc(test_input.nbytes)
d_output = cuda.mem_alloc(fp32_output.nbytes)
cuda.memcpy_htod(d_input, test_input)
fp32_context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(fp32_output, d_output)

# INT8 推論
int8_context = int8_engine.create_execution_context()
int8_output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_htod(d_input, test_input)
int8_context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(int8_output, d_output)


# ========== 計算差異 ==========
mse = np.mean((fp32_output - int8_output) ** 2)
max_abs_diff = np.max(np.abs(fp32_output - int8_output))
print(f"MSE (均方誤差): {mse:.6f}")
print(f"最大絕對誤差: {max_abs_diff:.6f}")
