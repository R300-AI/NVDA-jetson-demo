
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實作範例(1)：建立一個簡單的 CNN 模型，檢視模型結構與參數數量
適用平台：NVIDIA Jetson Orin（JetPack 6.2/6.2.1 + PyTorch for Jetson wheel）
功能：
  1) 定義 SmallCNN：輸入 3x32x32，輸出 10 類
  2) 可使用合成資料或 CIFAR-10
  3) 列印模型結構與參數總數/可訓練參數
  4) 在 GPU 可用時自動使用 CUDA
"""

import argparse
import os
import sys
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) 模型定義：SmallCNN
# -------------------------
class SmallCNN(nn.Module):
    """
    簡單兩層卷積 + MaxPool + 全連接：
      - conv1: 3 -> 32, kernel=3, padding=1
      - conv2: 32 -> 64, kernel=3, padding=1
      - pool: MaxPool2d(kernel_size=2)
      - fc:   64*16*16 -> 10
    對應輸入：N x 3 x 32 x 32
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 輸出 N x 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 輸出 N x 64 x 16 x 16（pool 後）
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)                # -> 32x16x16
        x = F.relu(self.conv2(x))       # -> 64x16x16

        x = x.flatten(1)                # -> N x (64*16*16)
        logits = self.fc(x)             # -> N x 10
        return logits


def count_params(m: nn.Module):
    """計算總參數數與可訓練參數數"""
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


# -------------------------
# 2) 測試資料：合成或 CIFAR-10
# -------------------------
def make_synthetic(batch_size: int = 64, n_classes: int = 10):
    """
    產生合成資料：RGB 32x32 影像與 10 類別標籤
    """
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, n_classes, (batch_size,))
    return x, y


def load_cifar10(batch_size: int = 64):
    """
    （選配）載入 CIFAR-10（需要 torchvision）
    """
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            "需要 torchvision 才能下載/載入 CIFAR-10；請安裝後重試：\n"
            "  pip3 install torchvision --extra-index-url https://download.pytorch.org/whl/cu118\n"
            f"原始錯誤：{e}"
        )

    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    x, y = next(iter(loader))
    return x, y


# -------------------------
# 3) 主程式：建立模型、檢視結構與參數
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_synthetic", action="store_true", help="使用合成資料")
    parser.add_argument("--use_cifar10", action="store_true", help="使用 CIFAR-10（需 torchvision）")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args([])

    if args.use_synthetic == args.use_cifar10:
        # 預設用合成資料；避免同時指定兩者
        args.use_synthetic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] torch version: {torch.__version__}")
    print(f"[Info] torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Info] CUDA device: {torch.cuda.get_device_name(0)}")

    # 建立模型
    model = SmallCNN(num_classes=10).to(device)

    # 檢視模型結構與參數數量
    total, trainable = count_params(model)
    print("\n===== Model Structure =====")
    print(model)
    print("\n===== Parameters =====")
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # 取一個 batch 資料做一次 forward（確認能跑）
    if args.use_synthetic:
        x, y = make_synthetic(batch_size=args.batch_size, n_classes=10)
        print("\n[Info] Using synthetic data.")
    else:
        x, y = load_cifar10(batch_size=args.batch_size)
        print("\n[Info] Using CIFAR-10 data.")

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    print(f"[Result] Input batch shape : {tuple(x.shape)}")
    print(f"[Result] Output logits     : {tuple(logits.shape)}")
    print(f"[Result] Pred sample (first 10): {pred[:10].tolist()}")
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 額外：列出簡單的參數分層與形狀（可幫助理解）
    print("\n===== Layer-wise Parameter Shapes =====")
    for n, p in model.named_parameters():
        print(f"{n:20s}  {tuple(p.shape)}  requires_grad={p.requires_grad}")


if __name__ == "__main__":
    main()
