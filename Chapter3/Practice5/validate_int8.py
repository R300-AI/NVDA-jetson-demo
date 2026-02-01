"""
Practice 5: INT8 量化精度驗證

執行方式:
    python3 validate_int8.py export
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8
    python3 validate_int8.py validate
"""

import sys
import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡單的 CNN"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        # TODO: 定義網路層
    
    def forward(self, x):
        # TODO: 實作前向傳播
        return x


if len(sys.argv) < 2:
    print("使用方式: python3 validate_int8.py [export|validate]")
    sys.exit(1)

mode = sys.argv[1]

if mode == "export":
    # TODO: 建立模型並匯出 ONNX
    pass

elif mode == "validate":
    # TODO: 載入 FP32 與 INT8 引擎，比較輸出差異
    pass

else:
    print(f"未知模式: {mode}")
