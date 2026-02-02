#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作範例：
CNN 模型的 Forward（前向傳播）與 Backward（反向傳播 + 參數更新）

功能：
1. 建立簡單 CNN（兩層 Conv + MaxPool）
2. 產生測試資料（合成資料）
3. 執行 Forward → 計算 loss → Backward → optimizer.step()
4. 印出每一步驟的結果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# 1) 定義 CNN 模型（簡化版，適合教學）
# ---------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 3→32
        self.pool = nn.MaxPool2d(2)                   # 32x32→16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32→64（仍 16x16）
        self.fc = nn.Linear(64 * 16 * 16, 10)         # 10 類別

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Conv1 + ReLU + Pool
        x = F.relu(self.conv2(x))              # Conv2 + ReLU
        x = x.flatten(1)
        return self.fc(x)

# ---------------------------------------------------------------
# 2) 產生測試資料（預設使用合成資料）
# ---------------------------------------------------------------
def synthetic_data(batch=64):
    x = torch.randn(batch, 3, 32, 32)
    y = torch.randint(0, 10, (batch,))
    return x, y

# ---------------------------------------------------------------
# 3) 主程式：forward + backward
# ---------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 建立模型
    model = SmallCNN().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # 測試資料
    x, y = synthetic_data()
    x, y = x.to(device), y.to(device)

    print("\n[1] Forward Pass ...")
    logits = model(x)
    print("Logits shape:", logits.shape)

    loss = criterion(logits, y)
    print("Loss:", loss.item())

    print("\n[2] Backward Pass ...")
    optimizer.zero_grad()
    loss.backward()  # 反向傳播（計算梯度）

    # 檢視某一層的梯度
    print("conv1.weight.grad mean:", model.conv1.weight.grad.mean().item())

    print("\n[3] Optimizer Step (SGD 更新參數)")
    optimizer.step()

    # 再跑一次 forward，看 loss 是否下降
    logits2 = model(x)
    loss2 = criterion(logits2, y)
    print("Loss after 1 step:", loss2.item())


if __name__ == "__main__":
    main()

