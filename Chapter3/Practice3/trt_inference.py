"""
Practice 3: TensorRT Python API 推論與 DLA 概念

題目說明:
1. 使用 Practice 1 或 Practice 2 產生的 .engine 檔案
2. 使用 TensorRT Python API 載入引擎並執行推論
3. 使用 --dumpLayerInfo 觀察層級部署資訊，了解 DLA 支援的運算子與 GPU Fallback 機制

注意: Jetson Orin Nano 未搭載 DLA，本練習以 GPU 模擬 DLA 的操作流程，
      讓你了解 DLA 支援的運算子限制與 Fallback 機制。

執行方式:
    python3 trt_inference.py

觀察層資訊（模擬 DLA 部署）:
    # 在有 DLA 的裝置上會使用: --useDLACore=0 --allowGPUFallback
    # Orin Nano 無 DLA，改用 GPU 並觀察層資訊
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpLayerInfo --exportLayerInfo=layers.json

DLA 支援的運算子（參考）:
    - Convolution (Conv)
    - Deconvolution (ConvTranspose)
    - Fully Connected (Gemm)
    - Pooling (MaxPool, AveragePool)
    - Activation (ReLU, Sigmoid, Tanh)
    - BatchNormalization
    - Scale
    - ElementWise (Add, Sub, Mul, Max, Min)
    - Concatenation

DLA 不支援的運算子（會 Fallback 到 GPU）:
    - Softmax
    - Resize (Upsample)
    - Split
    - Pad (某些模式)
    - ReduceMean, ReduceMax 等
"""

import tensorrt as trt
import numpy as np


def main():
    print("=" * 60)
    print("Practice 3: TensorRT Python API 推論")
    print("=" * 60)

    # 請先確認你有 Practice 1 產生的 simple_cnn_fp32.engine
    engine_path = "simple_cnn_fp32.engine"

    # ========== TODO 1: 載入 TensorRT 引擎 ==========
    """
    請使用 TensorRT API 載入引擎
    提示:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    """
    logger = None   # 請修改此行
    engine = None   # 請修改此行
    context = None  # 請修改此行


    # ========== TODO 2: 準備輸入輸出緩衝區 ==========
    """
    請準備 Host 與 Device 記憶體
    提示:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 定義形狀（SimpleCNN: 輸入 1x3x224x224，輸出 1x1000）
        input_shape = (1, 3, 224, 224)
        output_shape = (1, 1000)
        
        # Host 記憶體（CPU）
        h_input = np.random.randn(*input_shape).astype(np.float32)
        h_output = np.empty(output_shape, dtype=np.float32)
        
        # Device 記憶體（GPU）
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
    """
    h_input = None   # 請修改此行
    h_output = None  # 請修改此行
    d_input = None   # 請修改此行
    d_output = None  # 請修改此行


    # ========== TODO 3: 執行推論 ==========
    """
    請執行推論並取得結果
    提示:
        # 複製輸入到 GPU
        cuda.memcpy_htod(d_input, h_input)
        
        # 執行推論
        context.execute_v2([int(d_input), int(d_output)])
        
        # 複製輸出回 CPU
        cuda.memcpy_dtoh(h_output, d_output)
        
        print(f"預測類別: {np.argmax(h_output)}")
        print(f"信心分數: {np.max(h_output):.4f}")
    """


    print("\n" + "=" * 60)
    print("DLA 概念說明")
    print("=" * 60)
    print("\nJetson Orin Nano 未搭載 DLA，但你可以透過以下指令了解層資訊：")
    print("\n  trtexec --loadEngine=simple_cnn_fp32.engine \\")
    print("          --dumpLayerInfo --exportLayerInfo=layers.json")
    print("\n在有 DLA 的裝置（如 Orin NX、AGX Orin）上，可使用：")
    print("\n  trtexec --onnx=model.onnx --saveEngine=model_dla.engine \\")
    print("          --useDLACore=0 --allowGPUFallback \\")
    print("          --int8 --fp16")
    print("\nDLA 不支援的運算子會自動 Fallback 到 GPU 執行。")
    print("常見不支援的運算子: Softmax, Resize, Split, ReduceMean 等")


if __name__ == "__main__":
    main()
