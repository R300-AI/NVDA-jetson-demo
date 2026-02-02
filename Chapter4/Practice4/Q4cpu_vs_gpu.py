#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 CPU 與 GPU 訓練速度（PyTorch）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter


# -------------------------------------------------
# 1) 建立簡單 CNN
# -------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


# -------------------------------------------------
# 2) 合成資料
# -------------------------------------------------
def synthetic_data(batch=64):
    x = torch._____(batch, 3, 32, 32)        
    y = torch.randint(0, 10, (batch,))
    return x, y


# -------------------------------------------------
# 3) 單 epoch 訓練
# -------------------------------------------------
def train_one_epoch(model, device, use_amp=False):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn._____( )                 

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    x, y = synthetic_data()
    x, y = x.to(_____), y.to(_____)        

    t0 = perf_counter()

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)

        # TODO：AMP 自動混合精度
        with torch.cuda.amp.autocast(enabled=_____):   # True or False
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss)._____( )          
        scaler.step(optimizer)
        scaler.update()

    # TODO：若使用 CUDA，需同步以確保時間量測正確
    if device.type == "cuda":
        torch.cuda._____( )                 

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
    if torch.cuda._____( ):                  
        gpu_time = train_one_epoch(
            SmallCNN().to("cuda"),
            torch.device("cuda"),
            use_amp=_____                    # True or False
        )
        print(f"[GPU] Training time (AMP enabled): {gpu_time:.4f} sec")
    else:
        print("CUDA 不可用，無法比較 GPU 訓練")

    if torch.cuda.is_available():
        speedup = cpu_time / gpu_time
        print(f"\n===> Speedup: GPU 比 CPU 快 {speedup:.2f} 倍！")


if __name__ == "__main__":
    main()
