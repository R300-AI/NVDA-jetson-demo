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
        # === TODO: Conv1 + ReLU + Pool ===
        x = self.pool( F._____( self.conv1(x) ) )     # 提示：relu
        # === TODO: Conv2 + ReLU ===
        x = F._____( self.conv2(x) )                  # 提示：relu
        # === TODO: 展平後接全連接層 ===
        x = x._____(1)                                # 提示：flatten
        return self.fc(x)

# ---------------------------------------------------------------
# 2) 產生測試資料（預設使用合成資料）
# ---------------------------------------------------------------
def synthetic_data(batch=64):
    # === TODO: 隨機產生影像與標籤 ===
    x = torch._____(batch, 3, 32, 32)                 # 提示：randn
    y = torch.randint(0, 10, (batch,))
    return x, y

# ---------------------------------------------------------------
# 3) 主程式：forward + backward
# ---------------------------------------------------------------
def main():
    # === TODO: 選擇運算裝置（CUDA 可用時使用 GPU）===
    device = torch.device("cuda" if torch.cuda._____( ) else "cpu")  # 提示：is_available
    print("Using device:", device)

    # 建立模型
    model = SmallCNN().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=_____)        # 提示：1e-2

    # Loss function
    criterion = nn._____( )                                          # 提示：CrossEntropyLoss

    # 測試資料
    x, y = synthetic_data()
    x, y = x.to(device), y.to(device)

    print("\n[1] Forward Pass ...")
    # === TODO: 前向傳播取 logits ===
    logits = model(_____)                                            # 提示：x
    print("Logits shape:", logits.shape)

    # === TODO: 計算 loss ===
    loss = criterion(_____, _____)                                   # 提示：logits, y
    print("Loss:", loss.item())

    print("\n[2] Backward Pass ...")
    # === TODO: 反向前先將梯度清零 ===
    optimizer._____( )                                               # 提示：zero_grad
    # === TODO: 反向傳播（計算梯度）===
    loss._____( )                                                    # 提示：backward

    # 檢視某一層的梯度
    print("conv1.weight.grad mean:", model.conv1.weight.grad.mean().item())

    print("\n[3] Optimizer Step (SGD 更新參數)")
    # === TODO: 以 Optimizer 更新參數 ===
    optimizer._____( )                                               # 提示：step

    # 再跑一次 forward，看 loss 是否下降
    logits2 = model(x)
    loss2 = criterion(logits2, y)
    print("Loss after 1 step:", loss2.item())

    # （選配）如需更精準的 GPU 計時，可在這裡同步：
    # if device.type == "cuda":
    #     torch.cuda._____( )                                        # 提示：synchronize

if __name__ == "__main__":
    main()
