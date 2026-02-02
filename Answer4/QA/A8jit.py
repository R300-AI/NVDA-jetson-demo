## colab
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作：
測試 JIT 編譯後的推論效能（trace 與 script）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter


# -------------------------------------------------------
# 1. 測試模型：簡單 CNN（適合 trace 和 script）
# -------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 32x16x16
        x = F.relu(self.conv2(x))              # -> 64x16x16
        x = x.flatten(1)
        return self.fc(x)


# -------------------------------------------------------
# 2. 合成資料
# -------------------------------------------------------
def synthetic_data(batch=64):
    # === TODO: 產生隨機影像 tensor ===
    x = torch._____(batch, 3, 32, 32)         # randn
    return x


# -------------------------------------------------------
# 3. 推論效能量測（samples/sec）
# -------------------------------------------------------
@torch.inference_mode()
def bench_infer(model, device, iters=200, batch=64):
    model.eval().to(device)
    x = synthetic_data(batch).to(device)

    # 預熱，避免 Cold Start 影響
    for _ in range(10):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda._____( )                   # === TODO: synchronize

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda._____( )                   # === TODO: synchronize

    elapsed = perf_counter() - t0
    return iters * batch / elapsed


# -------------------------------------------------------
# 4. 主流程：Eager vs trace vs script
# -------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda._____( ) else "cpu")   # is_available
    print("Device:", device)

    model = SmallCNN()

    # ---------------- Eager 測速 ----------------
    fps_eager = bench_infer(model, device)
    print(f"[Eager ]  {fps_eager:8.2f} samples/sec")

    # ---------------- trace JIT ----------------
    example = torch.randn(1, 3, 32, 32).to(device)

    # === TODO: trace 編譯 ===
    traced = torch.jit._____( model.to(device).eval(), example )      # trace
    # === TODO: 儲存模型 ===
    torch.jit._____( traced, "model_traced.ts" )                      # save
    # === TODO: 載入模型 ===
    traced_loaded = torch.jit._____( "model_traced.ts" ).to(device)   # load

    fps_trace = bench_infer(traced_loaded, device)
    print(f"[trace ]  {fps_trace:8.2f} samples/sec")

    # ---------------- script JIT ----------------
    # === TODO: script 編譯 ===
    scripted = torch.jit._____( model.to(device).eval() )             # script
    # === TODO: 儲存模型 ===
    torch.jit._____( scripted, "model_scripted.ts" )                  # save
    # === TODO: 載入模型 ===
    scripted_loaded = torch.jit._____( "model_scripted.ts" ).to(device)  # load

    fps_script = bench_infer(scripted_loaded, device)
    print(f"[script]  {fps_script:8.2f} samples/sec")

    # （選配）印出部分 TorchScript IR
    try:
        print("\n[script] TorchScript code snippet:\n",
              scripted_loaded.code[:300], "...\n")
    except:
        pass


if __name__ == "__main__":
    main()
