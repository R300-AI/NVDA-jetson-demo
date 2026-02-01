"""
Practice 3: TensorRT Python API 推論

前置條件:
    需要 Practice 1 產生的 simple_cnn_fp32.engine（或 Practice 2 的 yolov8n 引擎）

執行方式:
    python3 trt_inference.py

觀察層級部署資訊:
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpLayerInfo

嘗試 DLA 編譯（Orin Nano 無 DLA，會出現 "Cannot create DLA engine" 錯誤）:
    trtexec --onnx=simple_cnn.onnx --useDLACore=0 --allowGPUFallback --dumpLayerInfo
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# ========== 設定參數 ==========
engine_path = "simple_cnn_fp32.engine"
input_shape = (1, 3, 224, 224)         # SimpleCNN 輸入形狀
output_shape = (1, 10)                 # SimpleCNN 輸出形狀（10 分類）

# 若使用 YOLOv8 引擎，請修改為:
# engine_path = "yolov8n_fp32.engine"
# input_shape = (1, 3, 640, 640)
# output_shape = (1, 84, 8400)         # YOLOv8n 輸出形狀

print(f"【Practice 3】載入引擎: {engine_path}")


# ========== TODO 1: 載入 TensorRT 引擎 ==========
# 提示: 參考 Chapter3 README「TensorRT Python API 推論」步驟一
# 1. 建立 trt.Logger
# 2. 用 open() 讀取 .engine 檔案
# 3. 用 trt.Runtime(logger).deserialize_cuda_engine() 反序列化
# 4. 用 engine.create_execution_context() 建立 context

logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()


# ========== TODO 2: 準備輸入輸出緩衝區 ==========
# 提示: 參考 Chapter3 README「TensorRT Python API 推論」步驟二
# Host (CPU) 端: 使用 np.random.randn() 產生輸入, np.empty() 準備輸出
# Device (GPU) 端: 使用 cuda.mem_alloc() 配置記憶體

h_input = np.random.randn(*input_shape).astype(np.float32)
h_output = np.empty(output_shape, dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)


# ========== TODO 3: 執行推論 ==========
# 提示: 參考 Chapter3 README「TensorRT Python API 推論」步驟三
# 1. cuda.memcpy_htod() 將輸入從 CPU 複製到 GPU
# 2. context.execute_v2() 執行推論
# 3. cuda.memcpy_dtoh() 將輸出從 GPU 複製到 CPU

cuda.memcpy_htod(d_input, h_input)
context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(h_output, d_output)


# ========== 輸出結果 ==========
print(f"輸入形狀: {input_shape}")
print(f"輸出形狀: {output_shape}")
# print(f"輸出結果: {h_output}")       # 取消註解以查看輸出
