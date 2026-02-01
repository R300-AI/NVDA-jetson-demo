"""
Practice 5: INT8 量化精度驗證

題目說明:
1. 使用 download_cifar10() 下載 CIFAR-10 測試集作為驗證資料
2. 使用 timm 建立 ResNet18 模型並匯出為 ONNX 格式
3. 分別編譯 FP32 與 INT8 兩個版本的 TensorRT 引擎
4. 使用 TensorRT Python API 載入兩個引擎，對 CIFAR-10 測試集進行推論
5. 比較 FP32 與 INT8 的 Top-1 準確率，分析量化對精度的影響

執行方式:
    # Step 1: 匯出 ONNX 模型
    python3 validate_int8.py --export

    # Step 2: 編譯 TensorRT 引擎
    trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_fp32.engine --shapes=input:1x3x32x32
    trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_int8.engine --shapes=input:1x3x32x32 --int8

    # Step 3: 執行驗證
    python3 validate_int8.py --validate
"""

import os
import pickle
import tarfile
import urllib.request
import argparse
import numpy as np
import torch
import timm


def download_cifar10(data_dir='./data'):
    """下載並解壓 CIFAR-10 資料集"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        print("下載 CIFAR-10 資料集...")
        urllib.request.urlretrieve(url, filename)
        print("解壓縮中...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(data_dir)
        os.remove(filename)
        print("完成!")
    
    return os.path.join(data_dir, 'cifar-10-batches-py')


def load_cifar10_test(data_dir):
    """載入 CIFAR-10 測試集"""
    test_path = os.path.join(data_dir, 'test_batch')
    with open(test_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    images = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(data_dict[b'labels'])
    
    # 標準化 (CIFAR-10 mean/std)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    images = (images - mean) / std
    
    return images.astype(np.float32), labels


def export_onnx():
    """匯出 ResNet18 ONNX 模型"""
    print("=" * 60)
    print("匯出 ResNet18 ONNX 模型")
    print("=" * 60)
    
    # ========== TODO 1: 載入 ResNet18 模型 ==========
    """
    請使用 timm 載入 ResNet18 並修改輸出類別數為 10
    提示:
        model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        model.eval()
    """
    model = None  # 請修改此行
    
    
    # ========== TODO 2: 匯出 ONNX ==========
    """
    請匯出 ONNX 模型（CIFAR-10 輸入為 32x32）
    提示:
        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(
            model,
            dummy_input,
            "resnet18_cifar10.onnx",
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
    """
    
    
    print("\n模型已匯出: resnet18_cifar10.onnx")
    print("\n下一步:")
    print("1. 編譯 FP32 引擎:")
    print("   trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_fp32.engine --shapes=input:1x3x32x32")
    print("\n2. 編譯 INT8 引擎:")
    print("   trtexec --onnx=resnet18_cifar10.onnx --saveEngine=resnet18_int8.engine --shapes=input:1x3x32x32 --int8")
    print("\n3. 執行驗證:")
    print("   python3 validate_int8.py --validate")


def validate_engines():
    """驗證 FP32 與 INT8 引擎的精度差異"""
    print("=" * 60)
    print("INT8 量化精度驗證")
    print("=" * 60)
    
    # 載入測試資料
    cifar_dir = download_cifar10('./data')
    test_images, test_labels = load_cifar10_test(cifar_dir)
    print(f"測試集大小: {len(test_labels)} 張圖片")
    
    # ========== TODO 3: 載入 TensorRT 引擎 ==========
    """
    請載入 FP32 與 INT8 兩個引擎
    提示:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger = trt.Logger(trt.Logger.WARNING)
        
        with open("resnet18_fp32.engine", "rb") as f:
            engine_fp32 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context_fp32 = engine_fp32.create_execution_context()
        
        with open("resnet18_int8.engine", "rb") as f:
            engine_int8 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context_int8 = engine_int8.create_execution_context()
    """
    context_fp32 = None  # 請修改此行
    context_int8 = None  # 請修改此行
    
    
    # ========== TODO 4: 執行推論並計算準確率 ==========
    """
    請對測試集執行推論並計算 Top-1 準確率
    提示:
        def run_inference(context, images):
            # 準備緩衝區
            h_output = np.empty((1, 10), dtype=np.float32)
            d_input = cuda.mem_alloc(images[0:1].nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            predictions = []
            for i in range(len(images)):
                h_input = images[i:i+1].astype(np.float32)
                cuda.memcpy_htod(d_input, h_input)
                context.execute_v2([int(d_input), int(d_output)])
                cuda.memcpy_dtoh(h_output, d_output)
                predictions.append(np.argmax(h_output))
            
            return np.array(predictions)
        
        # 計算準確率
        pred_fp32 = run_inference(context_fp32, test_images)
        pred_int8 = run_inference(context_int8, test_images)
        
        acc_fp32 = np.mean(pred_fp32 == test_labels) * 100
        acc_int8 = np.mean(pred_int8 == test_labels) * 100
        
        print(f"FP32 準確率: {acc_fp32:.2f}%")
        print(f"INT8 準確率: {acc_int8:.2f}%")
        print(f"精度差異: {acc_fp32 - acc_int8:.2f}%")
    """
    
    
    print("\n" + "=" * 60)
    print("分析結論")
    print("=" * 60)
    print("1. INT8 量化通常會造成 1-3% 的精度下降")
    print("2. 若精度下降過大，可考慮使用 Polygraphy 搭配真實校正資料")
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
