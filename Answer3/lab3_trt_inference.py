"""
Lab 3: TensorRT Python API 推論

執行方式:
    python3 lab3_trt_inference.py

觀察層資訊:
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpLayerInfo
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

print("TensorRT Python API 推論")

engine_path = "simple_cnn_fp32.engine"

# 載入引擎
logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
print(f"成功載入引擎: {engine_path}")

# 準備緩衝區
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

h_input = np.random.randn(*input_shape).astype(np.float32)
h_output = np.empty(output_shape, dtype=np.float32)

d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

# 執行推論
cuda.memcpy_htod(d_input, h_input)
context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(h_output, d_output)

print(f"預測類別: {np.argmax(h_output)}")
print(f"信心分數: {np.max(h_output):.4f}")
