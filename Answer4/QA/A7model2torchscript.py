## colab
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA Jetson Orin 實作：
將模型轉換為 TorchScript 格式（trace & script），並在 Orin 上做簡單推論效能比較
"""

import argparse
from time import perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1) 模型：兩層卷積 + MaxPool + 全連接（32x32 輸入；10 類）
#    - 結構固定：可用 trace
#    - 若要示範動態控制流，程式下方另有 DynamicHead 範例（需 script）
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)        # 32->16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x16x16
        x = F.relu(self.conv2(x))             # -> 64x16x16
        x = x.flatten(1)
        return self.fc(x)


# （選配）示範動態控制流（script 才能安全處理）
class DynamicHead(nn.Module):
    def __init__(self, base: nn.Module, th: float = 0.0):
        super().__init__()
        self.base = base
        self.th = th

    def forward(self, x):
        x = self.base.pool(F.relu(self.base.conv1(x)))
        x = F.relu(self.base.conv2(x))
        # 資料相依分支：只有當均值>th 才做微擾
        if x.mean() > self.th:
            x = x + 0.01 * torch.tanh(x)
        x = x.flatten(1)
        return self.base.fc(x)


# ------------------------------------------------------------
# 2) 合成資料 / （選配）CIFAR-10
# ------------------------------------------------------------
def synthetic_data(batch=64):
    x = torch.randn(batch, 3, 32, 32)
    y = torch.randint(0, 10, (batch, ))
    return x, y

def load_cifar10(batch=64):
    import torchvision
    from torchvision import transforms
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    return next(iter(loader))


# ------------------------------------------------------------
# 3) 推論效能（samples/sec）
# ------------------------------------------------------------
@torch.inference_mode()
def bench_infer(model, device, batch=64, iters=200):
    model.eval().to(device)
    x, _ = synthetic_data(batch)
    x = x.to(device)

    # 預熱
    for _ in range(10):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda._____( )            # === TODO：同步以準確量測（synchronize）

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda._____( )            # === TODO：同步以準確量測（synchronize）
    elapsed = perf_counter() - t0
    return iters * batch / elapsed     # samples/sec


# ------------------------------------------------------------
# 4) 主程式：轉換、儲存、載入與測速
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cifar10", action="store_true", help="改用 CIFAR-10 取一個 batch（需 torchvision）")
    parser.add_argument("--use_dynamic", action="store_true", help="使用含動態控制流的 DynamicHead（需要 script）")
    parser.add_argument("--trace_out", type=str, default="model_traced.ts", help="trace 輸出檔")
    parser.add_argument("--script_out", type=str, default="model_scripted.ts", help="script 輸出檔")
    parser.add_argument("--batch", type=int, default=64)
    # To prevent argparse from trying to parse kernel-specific arguments in Colab,
    # we pass an empty list to parse_args().
    args = parser.parse_args([])

    device = torch.device("cuda" if torch.cuda._____( ) else "cpu")  # === TODO：is_available
    print("Using device:", device)

    base = SmallCNN()
    model_eager = DynamicHead(base) if args.use_dynamic else base

    # （A）Eager 推論效能
    eager_fps = bench_infer(model_eager, device, batch=args.batch)
    print(f"[Eager]   {eager_fps:8.2f} samples/sec")

    # （B）trace（只適合結構固定模型；有動態分支請避免）
    if not args.use_dynamic:
        example = torch.randn(1, 3, 32, 32).to(device)
        # === TODO：以 trace 轉換並儲存 .ts 檔 ===
        traced = torch.jit._____( model_eager.to(device).eval(), example )  # trace
        torch.jit._____( traced, args.trace_out )                           # save
        print(f"[trace] 已輸出：{args.trace_out}")

        # === TODO：載入回測 ===
        traced_loaded = torch.jit._____( args.trace_out ).to(device)        # load
        traced_fps = bench_infer(traced_loaded, device, batch=args.batch)
        print(f"[trace]   {traced_fps:8.2f} samples/sec")
    else:
        print("[trace] 略過：DynamicHead 含資料相依分支，請使用 script")

    # （C）script（可保留 if/loop 等控制流）
    # === TODO：以 script 轉換、儲存並載入回測 ===
    scripted = torch.jit._____( model_eager.to(device).eval() )             # script
    torch.jit._____( scripted, args.script_out )                            # save
    print(f"[script] 已輸出：{args.script_out}")
    scripted_loaded = torch.jit._____( args.script_out ).to(device)         # load
    scripted_fps = bench_infer(scripted_loaded, device, batch=args.batch)
    print(f"[script]  {scripted_fps:8.2f} samples/sec")

    # 額外印出 script 的 IR（輔助教學）
    try:
        print("\n[script] TorchScript code:\n", scripted_loaded.code[:500], "...\n")
    except Exception:
        pass

    # （選配）取一個 batch 做正向，驗證輸出 shape
    if args.use_cifar10:
        x, _ = load_cifar10(batch=args.batch)
    else:
        x, _ = synthetic_data(batch=args.batch)
    x = x.to(device)
    with torch.inference_mode():
        y_eager = model_eager.to(device)(x)
        y_script = scripted_loaded(x)
    print("Output shapes ⇒ eager:", tuple(y_eager.shape), ", script:", tuple(y_script.shape))


if __name__ == "__main__":
    main()
