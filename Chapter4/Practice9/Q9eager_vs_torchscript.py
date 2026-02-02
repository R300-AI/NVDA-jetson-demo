## colab
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較原生 Eager 模型與 TorchScript 模型（trace/script）的推論效能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter


# ---------------------------------------------------------
# 1. 測試模型（固定結構，適合 trace/script）
# ---------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32→16
        x = F.relu(self.conv2(x))              # 保持 16x16
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------
# 2. 合成測試資料（避免 IO 影響效能）
# ---------------------------------------------------------
def synthetic_data(batch=64):
    # === TODO：產生隨機輸入 ===
    return torch._____(batch, 3, 32, 32)       


# ---------------------------------------------------------
# 3. 推論效能（samples/sec）
# ---------------------------------------------------------
@torch.inference_mode()
def bench(model, device, batch=64, iters=200):
    model.eval().to(device)
    x = synthetic_data(batch).to(device)

    # 預熱
    for _ in range(10):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda._____( )                   

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda._____( )                   

    elapsed = perf_counter() - t0
    return (batch * iters) / elapsed


# ---------------------------------------------------------
# 4. 主程式
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda._____( ) else "cpu")   
    print("Using device:", device)
    print("PyTorch:", torch.__version__)

    model = SmallCNN()

    # ===== Eager =====
    eager_fps = bench(model, device)
    print(f"[Eager   ] {eager_fps:10.2f} samples/sec")

    # ===== JIT: trace =====
    example = torch.randn(1, 3, 32, 32).to(device)

    # === TODO：trace 模型 ===
    traced = torch.jit._____( model.to(device), example )             
    # === TODO：儲存 traced 模型 ===
    torch.jit._____( traced, "smallcnn_traced.ts" )                  
    # === TODO：載入 traced 模型 ===
    traced_loaded = torch.jit._____( "smallcnn_traced.ts" ).to(device)  

    traced_fps = bench(traced_loaded, device)
    print(f"[Traced  ] {traced_fps:10.2f} samples/sec")

    # ===== JIT: script =====
    # === TODO：script 模型 ===
    scripted = torch.jit._____( model )                                
    # === TODO：儲存 scripted 模型 ===
    torch.jit._____( scripted, "smallcnn_scripted.ts" )
    # === TODO：載入 scripted 模型 ===
    scripted_loaded = torch.jit._____( "smallcnn_scripted.ts" ).to(device)

    scripted_fps = bench(scripted_loaded, device)
    print(f"[Scripted] {scripted_fps:10.2f} samples/sec")

    print("\n輸出檔案：smallcnn_traced.ts, smallcnn_scripted.ts")


if __name__ == "__main__":
    main()
