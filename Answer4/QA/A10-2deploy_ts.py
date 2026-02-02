## colab
# deploy_ts.py
import torch
# The SmallCNN class and bench_infer function are defined in this same cell,
# so they can be used directly without needing to import them.

def main():
    device = "cuda" if torch.cuda._____( ) else "cpu"  # === TODO: is_available
    print("Device:", device, "| PyTorch:", torch.__version__)

    # 1) 建立 eager 模型並量測基準
    eager = SmallCNN().to(device).eval()
    eager_fps = bench_infer(eager, device, iters=200, batch=64)
    print(f"[Eager   ] {eager_fps:10.2f} samples/sec")

    # 2) 轉 TorchScript（trace）
    example = torch.randn(1,3,32,32).to(device)
    traced = torch.jit._____(eager, example)           # === TODO: trace
    torch.jit._____(traced, "smallcnn_traced.ts")      # === TODO: save
    traced_loaded = torch.jit._____("smallcnn_traced.ts").to(device).eval()  # === TODO: load
    traced_fps = bench_infer(traced_loaded, device, iters=200, batch=64)
    print(f"[Traced  ] {traced_fps:10.2f} samples/sec")

    # 3) 轉 TorchScript（script）
    scripted = torch.jit._____(eager)                  # === TODO: script
    torch.jit._____(scripted, "smallcnn_scripted.ts")  # === TODO: save
    scripted_loaded = torch.jit._____("smallcnn_scripted.ts").to(device).eval()  # === TODO: load
    scripted_fps = bench_infer(scripted_loaded, device, iters=200, batch=64)
    print(f"[Scripted] {scripted_fps:10.2f} samples/sec")

    print("\n輸出檔案：smallcnn_traced.ts, smallcnn_scripted.ts")

if __name__ == "__main__":
    main()
