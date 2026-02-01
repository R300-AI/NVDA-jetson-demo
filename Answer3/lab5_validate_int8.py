"""
Lab 5 解答: INT8 量化精度驗證
執行: python3 lab5_validate_int8.py export   (匯出 ONNX)
     python3 lab5_validate_int8.py validate (驗證精度)
"""
import sys
import numpy as np
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(self.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(self.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 檢查命令列參數
if len(sys.argv) < 2:
    print("使用方式:")
    print("  python3 lab5_validate_int8.py export   # 匯出 ONNX")
    print("  python3 lab5_validate_int8.py validate # 驗證精度")
    sys.exit(1)

mode = sys.argv[1]

if mode == "export":
    # 匯出 ONNX
    model = SimpleCNN(num_classes=1000)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "simple_cnn.onnx", opset_version=17,
                      input_names=['input'], output_names=['output'])
    print("已匯出: simple_cnn.onnx")
    print("\n下一步編譯引擎:")
    print("  trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine")
    print("  trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8")

elif mode == "validate":
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 載入引擎
    logger = trt.Logger(trt.Logger.WARNING)
    with open("simple_cnn_fp32.engine", "rb") as f:
        engine_fp32 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    with open("simple_cnn_int8.engine", "rb") as f:
        engine_int8 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    ctx_fp32 = engine_fp32.create_execution_context()
    ctx_int8 = engine_int8.create_execution_context()
    
    # 推論函數
    def run_inference(context, images):
        h_output = np.empty((1, 1000), dtype=np.float32)
        d_input = cuda.mem_alloc(images[0:1].nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        outputs = []
        for i in range(len(images)):
            cuda.memcpy_htod(d_input, images[i:i+1])
            context.execute_v2([int(d_input), int(d_output)])
            cuda.memcpy_dtoh(h_output, d_output)
            outputs.append(h_output.copy())
        return np.array(outputs)
    
    # 執行驗證
    test_images = np.random.rand(100, 3, 224, 224).astype(np.float32)
    output_fp32 = run_inference(ctx_fp32, test_images)
    output_int8 = run_inference(ctx_int8, test_images)
    
    # 計算差異
    mse = np.mean((output_fp32 - output_int8) ** 2)
    max_diff = np.max(np.abs(output_fp32 - output_int8))
    agreement = np.mean(np.argmax(output_fp32, axis=-1) == np.argmax(output_int8, axis=-1)) * 100
    
    print(f"MSE: {mse:.6f}, Max Diff: {max_diff:.6f}, Agreement: {agreement:.2f}%")

else:
    print(f"未知模式: {mode}")
