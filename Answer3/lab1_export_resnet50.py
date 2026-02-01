"""
Practice 1: TensorRT 基本部署流程 - 匯出 ResNet-50 為 ONNX

題目說明:
1. 使用 timm 匯出 ResNet-50 模型為 ONNX 格式 (opset_version=17)
2. 使用 trtexec 將 ONNX 模型編譯成 FP32 的 TensorRT 引擎
3. 使用 trtexec 執行推論並加上 --dumpProfile 分析效能

執行方式:
    python3 export_resnet50.py

編譯 TensorRT 引擎:
    trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp32.engine --shapes=input:1x3x224x224

執行推論與效能分析:
    trtexec --loadEngine=resnet50_fp32.engine --dumpProfile --exportProfile=resnet50_profile.json
"""

import torch
import timm

def main():
    print("=" * 60)
    print("Practice 1: 匯出 ResNet-50 為 ONNX 格式")
    print("=" * 60)

    # ========== TODO 1: 載入預訓練的 ResNet-50 模型 ==========
    # 提示: model = timm.create_model('resnet50', pretrained=True)
    model = timm.create_model('resnet50', pretrained=True)


    # ========== TODO 2: 將模型設為評估模式 ==========
    # 提示: model.eval()
    model.eval()


    # ========== TODO 3: 建立假輸入 (1, 3, 224, 224) ==========
    # 提示: dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)


    # ========== TODO 4: 匯出 ONNX 模型 ==========
    onnx_path = "resnet50.onnx"
    """
    請使用 torch.onnx.export 匯出模型
    提示:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    """
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


    print(f"\n模型已匯出至: {onnx_path}")
    print("\n下一步:")
    print("1. 編譯 TensorRT 引擎:")
    print("   trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp32.engine --shapes=input:1x3x224x224")
    print("\n2. 執行推論與效能分析:")
    print("   trtexec --loadEngine=resnet50_fp32.engine --dumpProfile")

if __name__ == "__main__":
    main()
