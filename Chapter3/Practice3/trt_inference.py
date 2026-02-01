"""
Practice 3: TensorRT Python API 推論

執行方式:
    python3 trt_inference.py
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpLayerInfo
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

engine_path = "simple_cnn_fp32.engine"

# TODO 1: 載入 TensorRT 引擎


# TODO 2: 準備輸入輸出緩衝區


# TODO 3: 執行推論
