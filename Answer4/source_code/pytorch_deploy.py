## colab
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
將 PyTorch 模型部署到 Jetson Orin：
1) 以 PyTorch 匯出 ONNX
2) 直接呼叫 Python TensorRT library 載入 ONNX 並 build engine
3) 輸出模型變量（network I/O 與 engine 綁定）
4) 以 Python TensorRT 執行一次推論（顯示輸出 shape 與前幾個 logits）
"""

# Install TensorRT and PyCUDA if not already installed
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("Installing TensorRT and PyCUDA...")
    !pip install tensorrt pycuda
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # 建立 CUDA 上下文（注意：在單一進程內使用）

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 0) 測試用 PyTorch 模型（固定結構 CNN，適合 ONNX/TRT）
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)            # 32->16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # [N,32,16,16]
        x = F.relu(self.conv2(x))               # [N,64,16,16]
        x = x.flatten(1)
        return self.fc(x)


# -----------------------------
# 1) PyTorch → ONNX 匯出
# -----------------------------
def export_onnx(onnx_path="smallcnn.onnx", device="cuda"):
    model = SmallCNN().to(device).eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    torch.onnx.export(
        model, x, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print(f"[ONNX] 已匯出：{onnx_path}")


# -----------------------------
# 2) 以 Python TensorRT 載入 ONNX 並 build engine
#    - 支援 FP16（若平台支援）
#    - 設定動態 batch 的 OptProfile
# -----------------------------
def build_engine_from_onnx(onnx_path, fp16=True,
                           min_shape=(1, 3, 32, 32),
                           opt_shape=(64, 3, 32, 32),
                           max_shape=(256, 3, 32, 32)):
    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config  = builder.create_builder_config()
    parser  = trt.OnnxParser(network, logger)

    # 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[TensorRT] 解析 ONNX 失敗：")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    # 列印 Network 層級的 I/O（模型變量：inputs/outputs）
    print("\n[Network IO - from ONNX]")
    print("  num_inputs :", network.num_inputs)
    print("  num_outputs:", network.num_outputs)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"    [Input ] name={t.name}, dtype={t.dtype}, shape={t.shape}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"    [Output] name={t.name}, dtype={t.dtype}, shape={t.shape}")

    # 設定 Profile（動態 batch）
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TensorRT] 啟用 FP16")

    # 設定工作空間（依機台調整，這裡示範 512MB）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)

    # Build
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Build engine 失敗")
    print("[TensorRT] Engine 建置完成")

    return engine


# -----------------------------
# 3) 輸出 Engine 綁定（模型變量 in engine）
# -----------------------------
def print_engine_bindings(engine, context, batch=64):
    print("\n[Engine Bindings]")
    for i, name in enumerate(engine):
        is_input = engine.binding_is_input(name)
        dtype = engine.get_binding_dtype(name)
        if is_input:
            # 對輸入設定顯式 batch shape
            context.set_binding_shape(i, (batch, 3, 32, 32))
            shape = context.get_binding_shape(i)
        else:
            shape = context.get_binding_shape(i)  # 輸出 shape 需在 set_binding_shape 之後取得
        print(f"  idx={i:2d} | {'Input ' if is_input else 'Output'} | name={name:>12s} | dtype={dtype} | shape={tuple(shape)}")


# -----------------------------
# 4) 執行一次推論（Python TensorRT）
# -----------------------------
def run_inference(engine, context, batch=64):
    import numpy as np

    # 準備輸入資料（注意：dtype 需與 binding dtype 對齊）
    input_idx  = 0  # 本例只有一個輸入
    output_idx = 1  # 本例只有一個輸出
    in_dtype   = trt.nptype(engine.get_binding_dtype(input_idx))
    out_dtype  = trt.nptype(engine.get_binding_dtype(output_idx))

    context.set_binding_shape(input_idx, (batch, 3, 32, 32))
    out_shape = tuple(context.get_binding_shape(output_idx))

    host_in  = (np.random.randn(*((batch, 3, 32, 32))).astype(in_dtype))
    host_out = np.empty(out_shape, dtype=out_dtype)

    # 配置裝置記憶體
    d_in  = cuda.mem_alloc(host_in.nbytes)
    d_out = cuda.mem_alloc(host_out.nbytes)

    # H2D
    cuda.memcpy_htod(d_in, host_in)

    # 執行
    bindings = [int(d_in), int(d_out)]
    ok = context.execute_v2(bindings)
    if not ok:
        raise RuntimeError("TensorRT execute_v2 失敗")

    # D2H
    cuda.memcpy_dtoh(host_out, d_out)

    # 簡單列印幾個結果
    print("\n[Inference Result]")
    print("  output shape:", host_out.shape)
    print("  logits[0, :5]:", host_out[0, :5])

    # 釋放
    d_in.free()
    d_out.free()

    return host_out


# -----------------------------
# 5) 主程式
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| PyTorch:", torch.__version__, "| TensorRT:", trt.__version__)

    onnx_path = "smallcnn.onnx"
    if not os.path.exists(onnx_path):
        export_onnx(onnx_path, device=device)

    engine = build_engine_from_onnx(onnx_path, fp16=True)
    context = engine.create_execution_context()

    # 印出 Engine 綁定（模型變量）
    print_engine_bindings(engine, context, batch=64)

    # 做一次推論
    _ = run_inference(engine, context, batch=64)


if __name__ == "__main__":
    main()
