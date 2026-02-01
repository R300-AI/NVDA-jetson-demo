"""
Practice 1: TensorRT 基本部署流程

執行方式:
    python3 export_simple_cnn.py
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpProfile --exportProfile=simple_cnn_fp32_profile.json
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡單的 CNN 圖像分類器"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        # TODO: 定義網路層
    
    def forward(self, x):
        # TODO: 實作前向傳播
        return x


# 匯出 ONNX
model = SimpleCNN(num_classes=1000)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "simple_cnn.onnx", opset_version=17, input_names=['input'], output_names=['output'])
