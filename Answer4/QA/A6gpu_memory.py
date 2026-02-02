#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作：
檢查 GPU 記憶體使用狀況（PyTorch + 系統工具）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------
# 1. 建立簡單 CNN
# -----------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


# -----------------------------------------------------
# 2. 印出 GPU 記憶體
# -----------------------------------------------------
def print_mem(tag):
    alloc = torch.cuda._____( ) / 1024**2         # TODO：memory_allocated
    reserved = torch.cuda._____( ) / 1024**2      # TODO：memory_reserved
    print(f"[{tag}] allocated={alloc:.2f} MB, reserved={reserved:.2f} MB")


# -----------------------------------------------------
# 3. 主流程
# -----------------------------------------------------
def main():
    if not torch.cuda._____( ):                   # TODO：is_available
        print("CUDA 不可用，請確認 JetPack CUDA 安裝是否正常。")
        return

    device = torch.device("cuda")
    model = SmallCNN().to(_____)                  # TODO：device

    x = torch.randn(64, 3, 32, 32).to(_____)      # TODO：device
    y = torch.randint(0, 10, (64,)).to(_____)     # TODO：device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print("\n=== Jetson Orin GPU 記憶體測試 ===")

    print_mem("初始")

    # === TODO：前向傳播 ===
    logits = model(_____)                         # TODO：x
    print_mem("Forward 後")

    # === TODO：計算 Loss ===
    loss = criterion(_____, _____)                # TODO：logits, y

    # === TODO：反向傳播 ===
    loss._____( )                                 # TODO：backward
    print_mem("Backward 後")

    # === TODO：更新 Optimizer ===
    optimizer._____( )                            # TODO：step
    print_mem("Optimizer 更新後")

    # === TODO：釋放 GPU cache ===
    torch.cuda._____( )                           # TODO：empty_cache
    print_mem("empty_cache 後")


if __name__ == "__main__":
    main()
``
