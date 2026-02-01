"""
Practice 5: INT8 量化精度驗證

題目說明:
1. 使用純 PyTorch 自訂 SimpleCNN 並匯出為 ONNX 格式
2. 分別編譯 FP32 與 INT8 兩個版本的 TensorRT 引擎
3. 使用 TensorRT Python API 載入兩個引擎，對隨機測試資料進行推論
4. 比較 FP32 與 INT8 的輸出差異，分析量化對精度的影響

執行方式:
    # Step 1: 匯出 ONNX 模型
    python3 validate_int8.py --export

    # Step 2: 編譯 TensorRT 引擎
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine --shapes=input:1x3x224x224
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --shapes=input:1x3x224x224 --int8

    # Step 3: 執行驗證
    python3 validate_int8.py --validate
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡單的 CNN（僅使用 Conv2d、ReLU、MaxPool2d、Linear）"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # 224x224 -> 224x224
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 112x112 -> 112x112
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 56x56 -> 56x56
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(self.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(self.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def export_onnx():
    """匯出 SimpleCNN ONNX 模型"""
    print("=" * 60)
    print("匯出 SimpleCNN ONNX 模型")
    print("=" * 60)
    
    # 建立 SimpleCNN 模型
    model = SimpleCNN(num_classes=1000)
    model.eval()
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 匯出 ONNX（輸入為 224x224）
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "simple_cnn.onnx",
        opset_version=17,
        input_names=['input'],
        output_names=['output']
    )
    
    print("\n模型已匯出: simple_cnn.onnx")
    print("\n下一步:")
    print("1. 編譯 FP32 引擎:")
    print("   trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine --shapes=input:1x3x224x224")
    print("\n2. 編譯 INT8 引擎:")
    print("   trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --shapes=input:1x3x224x224 --int8")
    print("\n3. 執行驗證:")
    print("   python3 validate_int8.py --validate")


def validate_engines():
    """驗證 FP32 與 INT8 引擎的精度差異"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    print("=" * 60)
    print("INT8 量化精度驗證")
    print("=" * 60)
    
    # 產生隨機測試資料
    num_samples = 100
    print(f"產生 {num_samples} 筆隨機測試資料...")
    test_images = np.random.rand(num_samples, 3, 224, 224).astype(np.float32)
    
    # 載入 TensorRT 引擎
    logger = trt.Logger(trt.Logger.WARNING)
    
    print("\n載入 FP32 引擎...")
    with open("simple_cnn_fp32.engine", "rb") as f:
        engine_fp32 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context_fp32 = engine_fp32.create_execution_context()
    
    print("載入 INT8 引擎...")
    with open("simple_cnn_int8.engine", "rb") as f:
        engine_int8 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context_int8 = engine_int8.create_execution_context()
    
    def run_inference(context, images):
        """對所有圖片執行推論"""
        h_output = np.empty((1, 1000), dtype=np.float32)
        d_input = cuda.mem_alloc(images[0:1].nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        outputs = []
        for i in range(len(images)):
            h_input = images[i:i+1].astype(np.float32)
            cuda.memcpy_htod(d_input, h_input)
            context.execute_v2([int(d_input), int(d_output)])
            cuda.memcpy_dtoh(h_output, d_output)
            outputs.append(h_output.copy())
            
            if (i + 1) % 20 == 0:
                print(f"  已處理 {i + 1} / {len(images)} 筆")
        
        return np.array(outputs)
    
    # 執行推論
    print("\n執行 FP32 推論...")
    output_fp32 = run_inference(context_fp32, test_images)
    
    print("\n執行 INT8 推論...")
    output_int8 = run_inference(context_int8, test_images)
    
    # 計算輸出差異
    mse = np.mean((output_fp32 - output_int8) ** 2)
    max_diff = np.max(np.abs(output_fp32 - output_int8))
    pred_fp32 = np.argmax(output_fp32, axis=-1)
    pred_int8 = np.argmax(output_int8, axis=-1)
    agreement = np.mean(pred_fp32 == pred_int8) * 100
    
    print("\n" + "=" * 60)
    print("驗證結果")
    print("=" * 60)
    print(f"均方誤差 (MSE):     {mse:.6f}")
    print(f"最大絕對誤差:       {max_diff:.6f}")
    print(f"預測一致性:         {agreement:.2f}%")
    
    print("\n" + "=" * 60)
    print("分析結論")
    print("=" * 60)
    if agreement >= 99.0:
        print("✓ INT8 量化對此模型影響極小，建議採用 INT8 以獲得更高效能")
    elif agreement >= 95.0:
        print("△ INT8 量化造成輕微差異，請根據應用場景決定是否採用")
    else:
        print("✗ INT8 量化造成明顯差異，建議使用 Polygraphy 搭配真實校正資料")
        print("  或考慮使用 FP16 精度")


def main():
    parser = argparse.ArgumentParser(description='INT8 量化精度驗證')
    parser.add_argument('--export', action='store_true', help='匯出 ONNX 模型')
    parser.add_argument('--validate', action='store_true', help='驗證引擎精度')
    args = parser.parse_args()
    
    if args.export:
        export_onnx()
    elif args.validate:
        validate_engines()
    else:
        print("使用方式:")
        print("  python3 validate_int8.py --export    # 匯出 ONNX 模型")
        print("  python3 validate_int8.py --validate  # 驗證引擎精度")


if __name__ == "__main__":
    main()
