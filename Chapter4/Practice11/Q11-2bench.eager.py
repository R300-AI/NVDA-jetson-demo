## colab
# bench_eager.py
import torch

def main_eager():
    device = "cuda" if torch.cuda._____( ) else "cpu"    
    print("Device:", device, "| PyTorch:", torch.__version__)

    model = SmallCNN()
    fps = bench_infer(model, device=device, iters=200, batch=64)
    print(f"[PyTorch Eager] {fps:10.2f} samples/sec")
``
