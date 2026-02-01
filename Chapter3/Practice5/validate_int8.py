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
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8

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
    
    # ========== TODO 1: 建立 SimpleCNN 模型 ==========
    """
    請建立 SimpleCNN 模型
    提示:
        model = SimpleCNN(num_classes=1000)
        model.eval()
    """
    model = None  # 請修改此行
    
    
    # ========== TODO 2: 匯出 ONNX ==========
    """
    請匯出 ONNX 模型（輸入為 224x224）
    提示:
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            "simple_cnn.onnx",
            opset_version=17,
            input_names=['input'],
            output_names=['output']
        )
    """
    
    
    print("\n模型已匯出: simple_cnn.onnx")
    print("\n下一步:")
    print("1. 編譯 FP32 引擎:")
    print("   trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine")
    print("\n2. 編譯 INT8 引擎:")
    print("   trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_int8.engine --int8")
    print("\n3. 執行驗證:")
    print("   python3 validate_int8.py --validate")


def validate_engines():
    """驗證 FP32 與 INT8 引擎的精度差異"""
    print("=" * 60)
    print("INT8 量化精度驗證")
    print("=" * 60)
    
    # 產生隨機測試資料
    num_samples = 100
    print(f"產生 {num_samples} 筆隨機測試資料...")
    test_images = np.random.rand(num_samples, 3, 224, 224).astype(np.float32)
    
    # ========== TODO 3: 載入 TensorRT 引擎並執行推論 ==========
    """
    請載入 FP32 與 INT8 兩個引擎並比較輸出差異
    提示:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger = trt.Logger(trt.Logger.WARNING)
        
        # 載入引擎
        with open("simple_cnn_fp32.engine", "rb") as f:
            engine_fp32 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context_fp32 = engine_fp32.create_execution_context()
        
        # 執行推論並比較輸出差異...
    """
    
    
    print("\n" + "=" * 60)
    print("分析結論")
    print("=" * 60)
    print("1. INT8 量化通常會造成輕微的精度差異")
    print("2. 若差異過大，可考慮使用 Polygraphy 搭配真實校正資料")
    print("3. 對於精度敏感的應用，建議使用 FP16 而非 INT8")


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
