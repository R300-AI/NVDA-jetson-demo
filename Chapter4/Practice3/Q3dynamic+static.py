#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 教學：
(一) PyTorch 動態計算（Eager, dynamic control flow）
(二) PyTorch 靜態計算（TorchScript: script 與 trace）

重點：
- 動態：以 tensor 數值決定分支/迴圈（if/for）=> 研究/除錯友善、彈性高
- 靜態：將模型"凍結"為可序列化/可優化的 graph（TorchScript）=> 部署/跨語言推論友善
- 示範 trace（易出錯於 data-dependent branch） vs script（能保留控制流）
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1) 模型定義：DynamicCNN 與 SimpleCNN
# =========================================================

class DynamicCNN(nn.Module):
    """
    動態特徵：
      - 以輸入張量的平均值（data-dependent）決定是否走 conv2 分支
      - 以輸入張量的 L2 norm 近似值決定迴圈次數（data-dependent loop）
    結果：Eager 模式下可正常運作；trace 可能漏掉分支；script 能正確保留控制流
    """
    def __init__(self, use_loop: bool = True):
        super().__init__()
        self.use_loop = use_loop # Re-adding this line
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一層固定
        x = F.relu(self.conv1(x)) # (N, 32, 32, 32)
        x = self.pool(x)           # 32x32 -> 16x16. Moved pooling here

        # 動態分支：依據資料的平均值決定是否再做 conv2
        if x.mean() > 0:           # data-dependent condition
            x = F.relu(self.conv2(x)) # Now conv2 operates on (N, 32, 16, 16)

        # 動態迴圈（可關閉）：依 L2 norm 決定重複次數（最多3次）
        if self.use_loop:
            n = int(torch.clamp(x.norm().detach() // 1000, min=0, max=3).item())
            for _ in range(n):
                x = x + 0.01 * torch.tanh(x)  # 微小非線性擾動

        x = x.flatten(1)
        return self.head(x)


class SimpleCNN(nn.Module):
    """
    無資料相依的控制流（適合 trace）
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x16x16
        x = F.relu(self.conv2(x))             # -> 64x16x16
        x = x.flatten(1)
        return self.fc(x)


# =========================================================
# 2) 測試資料：合成 or CIFAR-10（選配）
# =========================================================
def synthetic_data(batch=64):
    x = torch.randn(batch, 3, 32, 32)
    y = torch.randint(0, 10, (batch,))
    return x, y

def load_cifar10(batch=64):
    import torchvision
    from torchvision import transforms
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    return next(iter(loader))


# =========================================================
# 3) 動態 vs 靜態：示範流程
# =========================================================
def run_dynamic(device, use_cifar):
    print("\n=== (一) 動態計算：Eager（含資料相依分支/迴圈） ===")
    # === TODO: 模型移動到裝置並切換為 eval 模式 ===
    model = DynamicCNN(use_loop=True).to(_____) .eval()      
    x, y = (load_cifar10() if use_cifar else synthetic_data())
    x = x.to(device)

    # === TODO: 以推論模式執行 forward ===
    with torch._____( ):                                     
        out = model(x)
    print("Dynamic Eager output shape:", tuple(out.shape))
    print("Eager example logits[0][:5]:", out[0, :5].tolist())


def run_static(device, use_cifar):
    print("\n=== (二) 靜態計算：TorchScript ===")
    # A) script 適合有 control flow 的模型
    dyn_model = DynamicCNN(use_loop=True).to(device).eval()
    # === TODO: 使用 torch.jit.script 以保留 if/loop 控制流 ===
    scripted = torch.jit._____(dyn_model)                   
    x, _ = (load_cifar10() if use_cifar else synthetic_data())
    x = x.to(device)
    with torch.inference_mode():
        out_s = scripted(x)
    print("[script] output shape:", tuple(out_s.shape))
    print("[script] example logits[0][:5]:", out_s[0, :5].tolist())

    # 顯示 TorchScript code（可看到 if/loop 被保留在 IR 中）
    try:
        print("\n[script] TorchScript code:\n", scripted.code)
    except Exception:
        pass

    # B) trace 適用於沒有資料相依控制流的模型（作為對比）
    print("\n--- trace 對比（適合無控制流的 SimpleCNN） ---")
    simple = SimpleCNN().to(device).eval()
    ex = torch.randn(1, 3, 32, 32, device=device)
    # === TODO: 使用 torch.jit.trace 建立靜態路徑 ===
    traced = torch.jit._____(simple, ex)                     
    with torch.inference_mode():
        out_t = traced(x)
    print("[trace] output shape:", tuple(out_t.shape))
    print("[trace] example logits[0][:5]:", out_t[0, :5].tolist())

    # 注意：若用 trace 對有 if/loop 的 DynamicCNN，trace 只會「記錄」當下走過的分支，
    # 可能漏掉另一個分支/不同迴圈次數，導致不可預期；因此這裡不建議。


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cifar10", action="store_true",
                        help="使用 CIFAR-10，需 torchvision；預設使用合成資料")
    args = parser.parse_args([]) # Modified this line

    # === TODO: 選擇運算裝置（CUDA 可用時用 GPU）===
    device = torch.device("cuda" if torch.cuda._____( ) else "cpu")  
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    run_dynamic(device, use_cifar=args.use_cifar10)
    run_static(device, use_cifar=args.use_cifar10)


if __name__ == "__main__":
    main()
``
