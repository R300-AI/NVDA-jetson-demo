# export_onnx.py
import torch

# Install onnxscript if not installed
try:
    import onnxscript
except ImportError:
    print("Installing onnxscript...")
    !pip install onnxscript
    import onnxscript

def main_onnx():
    device = "cuda" if torch.cuda._____( ) else "cpu"   # === TODO: is_available
    model = SmallCNN().to(device).eval()
    x = torch.randn(1, 3, 32, 32).to(device)

    # === TODO: 將模型匯出為 ONNX ===
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
    main_eager()
    main_onnx()
