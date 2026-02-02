
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
# 1. 建立簡單 CNN，供記憶體分配測試
# -----------------------------------------------------
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


# -----------------------------------------------------
# 2. 印出 GPU 記憶體
# -----------------------------------------------------
def print_mem(tag):
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] allocated={alloc:.2f} MB, reserved={reserved:.2f} MB")


# -----------------------------------------------------
# 3. 主流程：前向、反向時記憶體變化
# -----------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，請確認 JetPack 6.2 CUDA 12.6 是否正常。")
        return

    device = torch.device("cuda")
    model = SmallCNN().to(device)
    x = torch.randn(64, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (64,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print("\n=== Jetson Orin GPU 記憶體測試 ===")

    print_mem("初始")
    logits = model(x)
    print_mem("Forward 後")

    loss = criterion(logits, y)
    loss.backward()
    print_mem("Backward 後")

    optimizer.step()
    print_mem("Optimizer 更新後")

    # 釋放 GPU cache
    torch.cuda.empty_cache()
    print_mem("empty_cache 後")


if __name__ == "__main__":
    main()
