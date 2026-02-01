"""
Practice 3 輔助模板: 檢查 ONNX 模型的 DLA 支援情況

執行方式:
    python3 check_dla_support.py
"""

import onnx

# ========== TODO: 設定 ONNX 模型路徑 ==========
onnx_path = "simple_cnn.onnx"          # 請修改為你的模型路徑


# ========== DLA 支援的運算子清單 ==========
# 參考: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay

DLA_SUPPORTED_OPS = {
    "Conv", "ConvTranspose",
    "MaxPool", "AveragePool", "GlobalAveragePool",
    "Relu", "Sigmoid", "Tanh", "LeakyRelu",
    "BatchNormalization",
    "Add", "Sub", "Mul", "Div",
    "Resize", "Upsample",
    "Concat",
    "Gemm", "MatMul",
}

DLA_UNSUPPORTED_OPS = {
    "Flatten": "改變 tensor 形狀",
    "Reshape": "改變 tensor 形狀",
    "Transpose": "維度轉置",
    "Squeeze": "移除維度",
    "Unsqueeze": "增加維度",
    "Softmax": "複雜運算",
    "Slice": "切片操作",
    "Split": "分割操作",
    "Gather": "索引操作",
    "Shape": "形狀操作",
    "Cast": "型別轉換",
    "Pad": "填充操作",
}


# ========== 分析模型 ==========
print(f"\n{'='*50}")
print(f"DLA 支援分析: {onnx_path}")
print(f"{'='*50}\n")

model = onnx.load(onnx_path)

op_counts = {}
dla_supported = []
dla_unsupported = []

for node in model.graph.node:
    op_type = node.op_type
    op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    if op_type in DLA_SUPPORTED_OPS:
        dla_supported.append((node.name, op_type))
    elif op_type in DLA_UNSUPPORTED_OPS:
        dla_unsupported.append((node.name, op_type))

# 輸出統計
print("【運算子統計】")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    status = "✓" if op in DLA_SUPPORTED_OPS else "✗"
    print(f"  {status} {op}: {count} 個")

print(f"\n【需要 GPU Fallback】({len(dla_unsupported)} 層)")
for name, op in dla_unsupported:
    reason = DLA_UNSUPPORTED_OPS.get(op, "")
    print(f"  ✗ {op}: {reason}")

# 總結
total = len(model.graph.node)
print(f"\n【評估結果】")
print(f"  DLA 可執行: {len(dla_supported)}/{total} ({len(dla_supported)/total*100:.1f}%)")
print(f"  需 GPU Fallback: {len(dla_unsupported)}/{total}")
