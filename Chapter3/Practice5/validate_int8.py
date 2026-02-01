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
    """簡單的 CNN（與 Practice 1 相同）"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # ========== TODO 1: 定義網路層 ==========
        # 提示: 需與 Practice 1 的模型結構一致
        
        self.features = None           # 請定義卷積層
        self.classifier = None         # 請定義全連接層
    
    def forward(self, x):
        # ========== TODO 2: 實作前向傳播 ==========
        return x                       # 請修改此行


def export_onnx():
    """匯出模型為 ONNX"""
    print("【Practice 5】匯出 SimpleCNN 為 ONNX")
    
    model = SimpleCNN(num_classes=10)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "simple_cnn.onnx",
        opset_version=17,
        input_names=['input'],
        output_names=['output']
    )
    print("已匯出: simple_cnn.onnx")
    print("下一步:")
    print("  trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine")
    print("  trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8")


def validate():
    """比較 FP32 與 INT8 的輸出差異"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    print("【Practice 5】驗證 INT8 量化精度")
    
    # ========== TODO 3: 載入引擎 ==========
    # 提示: 參考 Chapter3 README「TensorRT Python API 推論」
    
    fp32_engine_path = "simple_cnn_fp32.engine"
    int8_engine_path = "simple_cnn_int8.engine"
    
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(fp32_engine_path, "rb") as f:
        fp32_engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    with open(int8_engine_path, "rb") as f:
        int8_engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    # ========== TODO 4: 準備緩衝區並執行推論 ==========
    # 提示: 參考 Practice 3 的推論流程
    
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 請完成 FP32 與 INT8 的推論，並將結果存入 fp32_output, int8_output
    fp32_output = None                 # 請完成 FP32 推論
    int8_output = None                 # 請完成 INT8 推論
    
    # ========== 計算差異 ==========
    # mse = np.mean((fp32_output - int8_output) ** 2)
    # max_abs_diff = np.max(np.abs(fp32_output - int8_output))
    # print(f"MSE (均方誤差): {mse:.6f}")
    # print(f"最大絕對誤差: {max_abs_diff:.6f}")


# ========== 主程式 ==========
mode = sys.argv[1] if len(sys.argv) > 1 else None

if mode == "export":
    export_onnx()
elif mode == "validate":
    validate()
else:
    print("使用方式:")
    print("  python3 validate_int8.py export    # 匯出 ONNX")
    print("  python3 validate_int8.py validate  # 驗證精度")
