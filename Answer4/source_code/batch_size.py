#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作：
測試不同 Batch Size 對 GPU 訓練效能的影響
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter


# ---------------------------------------------------------
# 1) 模型：簡單 CNN（用於效能測試）
# ---------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------
# 2) 產生測試資料
# ---------------------------------------------------------
def synthetic_data(batch):
    x = torch.randn(batch, 3, 32, 32)
    y = torch.randint(0, 10, (batch,))
    return x, y


# ---------------------------------------------------------
# 3) 測試不同 Batch Size 的吞吐量
# ---------------------------------------------------------
def measure_throughput(batch_sizes=[16, 32, 64, 128, 256]):
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot measure GPU throughput.")
        return {} # Return empty results if CUDA is not available

    device = torch.device("cuda")
    model = SmallCNN().to(device).eval()

    results = {}

    for bs in batch_sizes:
        x, _ = synthetic_data(bs)
        x = x.to(device)

        iters = 200
        torch.cuda.synchronize()
        t0 = perf_counter()

        for _ in range(iters):
            y = model(x)

        torch.cuda.synchronize()
        elapsed = perf_counter() - t0

        throughput = (iters * bs) / elapsed  # samples/sec
        results[bs] = throughput

        print(f"BS={bs:4d}  throughput={throughput:8.2f} samples/sec")

    return results


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    print("\n=== Jetson Orin Batch Size 吞吐量測試 ===")
    batch_sizes = [16, 32, 64, 128, 256, 512]
    results = measure_throughput(batch_sizes)

    print("\n=== 結果總結 ===")
    for bs, thr in results.items():
        print(f"Batch Size {bs:4d} → {thr:10.2f} samples/sec")


if __name__ == "__main__":
    main()
