#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作：
比較 CPU 與 GPU 訓練速度（PyTorch）

流程：
1. 建立簡單 CNN
2. 產生合成資料
3. CPU 訓練 1 epoch
4. GPU 訓練 1 epoch（含 AMP，可提升效能）
5. 輸出訓練時間比較
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter


# -------------------------------------------------
# 1) 建立簡單 CNN（32x32, 10 類別）
# -------------------------------------------------
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


# -------------------------------------------------
# 2) 測試資料（合成）：64 張 3x32x32 影像
# -------------------------------------------------
def synthetic_data(batch=64):
    x = torch.randn(batch, 3, 32, 32)
    y = torch.randint(0, 10, (batch,))
    return x, y


# -------------------------------------------------
# 3) 單次訓練迴圈
# -------------------------------------------------
def train_one_epoch(model, device, use_amp=False):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    x, y = synthetic_data()
    x, y = x.to(device), y.to(device)

    t0 = perf_counter()

    for _ in range(100):   # 100 個 iteration
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return perf_counter() - t0


# -------------------------------------------------
# 4) 主程式
# -------------------------------------------------
def main():
    print("PyTorch:", torch.__version__)

    # CPU 測試
    cpu_time = train_one_epoch(SmallCNN().to("cpu"), torch.device("cpu"))
    print(f"[CPU] Training time: {cpu_time:.4f} sec")

    # GPU 測試（Jetson Orin）
    if torch.cuda.is_available():
        gpu_time = train_one_epoch(
            SmallCNN().to("cuda"), torch.device("cuda"), use_amp=True
        )
        print(f"[GPU] Training time (AMP enabled): {gpu_time:.4f} sec")
    else:
        print("CUDA 不可用，無法比較 GPU 訓練")

    if torch.cuda.is_available():
        speedup = cpu_time / gpu_time
        print(f"\n===> Speedup: GPU 比 CPU 快 {speedup:.2f} 倍！")


if __name__ == "__main__":
    main()
