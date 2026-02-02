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
    x = torch.randn(batch, 3, 32, 32)
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
        torch.cuda.synchronize()

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

