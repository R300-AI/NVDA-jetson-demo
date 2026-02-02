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
        self.pool  = nn.MaxPool2d(2)            # 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc    = nn.Linear(64*16*16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # [N,32,16,16]
        x = F.relu(self.conv2(x))                # [N,64,16,16]
        x = x.flatten(1)
        return self.fc(x)

def make_synth(batch=64):
    # 使用合成資料避免 IO 對效能的影響
    x = torch.randn(batch, 3, 32, 32)
    return x

@torch.inference_mode()
def bench_infer(model, device="cuda", iters=200, batch=64):
    """回傳 samples/sec，並以 synchronize 確保 GPU 計時正確"""
    model.eval().to(device)
    x = make_synth(batch).to(device)

    # 預熱，避免 cold-start 影響
    for _ in range(10):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    return iters * batch / (perf_counter() - t0)

# bench_eager.py
import torch
# SmallCNN and bench_infer are defined in this cell, no import needed.

def main_eager():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| PyTorch:", torch.__version__)

    model = SmallCNN()
    fps = bench_infer(model, device=device, iters=200, batch=64)
    print(f"[PyTorch Eager] {fps:10.2f} samples/sec")

# export_onnx.py
import torch
# SmallCNN and bench_infer are defined in this cell, no import needed.

# Install onnxscript if not already installed
try:
    import onnxscript
except ImportError:
    print("Installing onnxscript...")
    !pip install onnxscript
    import onnxscript

def main_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device).eval()
    x = torch.randn(1, 3, 32, 32).to(device)

    torch.onnx.export(
        model, x, "smallcnn.onnx",
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print("已輸出：smallcnn.onnx")

    bash_script = """
set -euo pipefail

ONNX=smallcnn.onnx
PLAN=smallcnn_fp16.plan
SHAPES=input:64x3x32x32

echo "[1] 以 FP16 生成 TensorRT 引擎（.plan）"
trtexec --onnx=${ONNX} --saveEngine=${PLAN} --fp16 --shapes=${SHAPES} \
        --useCudaGraph --noDataTransfers

echo "[2] 載入引擎做效能測試（200 次迭代，回報 P95）"
trtexec --loadEngine=${PLAN} --iterations=200 --percentile=95 \
        --useCudaGraph --noDataTransfers
"""
    print("\nExecuting TensorRT commands...")
    get_ipython().system_raw(bash_script)

if __name__ == "__main__":
    # Run both main functions sequentially
    main_eager()
    main_onnx()
