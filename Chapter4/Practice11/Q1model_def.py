## colab

# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter

class SmallCNN(nn.Module):
    """32x32 RGB -> 10 classes"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64*16*16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)

def make_synth(batch=64):
    # === TODO: 建立合成資料，避免 IO 干擾效能 ===
    x = torch._____(batch, 3, 32, 32)          
    return x

@torch.inference_mode()
def bench_infer(model, device="cuda", iters=200, batch=64):
    """回傳 samples/sec，並以 synchronize 確保 GPU 計時正確"""
    model.eval().to(device)
    x = make_synth(batch).to(device)

    # 預熱
    for _ in range(10):
        _ = model(x)
    if device == "cuda":
        torch.cuda._____( )                  

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda._____( )                  # === TODO: synchronize

    return iters * batch / (perf_counter() - t0)
