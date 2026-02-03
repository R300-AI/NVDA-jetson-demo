# lab3_trt_inference_cpu.py
"""
CPU-only 版 lab3_trt_inference.py
使用 PyTorch 或 ONNX Runtime CPU 做推論
完全不需要 GPU / TensorRT / PyCUDA
"""

import os
import numpy as np

# 選擇推論方式：'onnx' 或 'pytorch'
INFERENCE_MODE = 'onnx'  # 或 'pytorch'

# -------- PyTorch CPU 範例 --------
if INFERENCE_MODE == 'pytorch':
    import torch
    from lab1_export_simple_cnn import SimpleCNN  # 或你的模型檔案

    # 初始化模型
    model = SimpleCNN()
    model_path = "simple_cnn.pth"  # 請確認模型檔案路徑
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 模擬輸入
    x = torch.randn(1, 3, 224, 224)

    # CPU 推論
    with torch.no_grad():
        y = model(x)

    print("PyTorch CPU 推論完成，結果 shape:", y.shape)
    print(y)

# -------- ONNX Runtime CPU 範例 --------
elif INFERENCE_MODE == 'onnx':
    import onnxruntime as ort

    onnx_model_path = "simple_cnn.onnx"  # 或你的 ONNX 模型
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"找不到 ONNX 模型: {onnx_model_path}")

    # 初始化 ONNX Runtime Session（CPU）
    ort_session = ort.InferenceSession(onnx_model_path)

    # 模擬輸入
    # ONNX 輸入名稱通常可以透過 ort_session.get_inputs() 確認
    input_name = ort_session.get_inputs()[0].name
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # CPU 推論
    outputs = ort_session.run(None, {input_name: x})

    print("ONNX Runtime CPU 推論完成，結果 shape:", outputs[0].shape)
    print(outputs[0])

else:
    raise ValueError("INFERENCE_MODE 必須是 'onnx' 或 'pytorch'")
