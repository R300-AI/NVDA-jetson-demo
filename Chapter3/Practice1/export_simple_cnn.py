"""
Practice 1: TensorRT 基本部署流程 - 匯出自訂 CNN 為 ONNX

題目說明:
1. 使用純 PyTorch 自訂 CNN 分類器並匯出為 ONNX 格式 (opset_version=17)
2. 使用 trtexec 將 ONNX 模型編譯成 FP32 的 TensorRT 引擎
3. 使用 trtexec 執行推論並加上 --dumpProfile 分析效能

執行方式:
    python3 export_simple_cnn.py

編譯 TensorRT 引擎:
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine

執行推論與效能分析:
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpProfile --exportProfile=simple_cnn_profile.json
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡單的 CNN 圖像分類器（僅使用 Conv2d、ReLU、MaxPool2d、Linear）"""
    
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
        x = x.view(x.size(0), -1)                # Flatten
        x = self.fc(x)
        return x


def main():
    print("=" * 60)
    print("Practice 1: 匯出自訂 CNN 為 ONNX 格式")
    print("=" * 60)

    # ========== TODO 1: 建立 SimpleCNN 模型 ==========
    # 提示: model = SimpleCNN(num_classes=1000)
    model = SimpleCNN(num_classes=1000)


    # ========== TODO 2: 將模型設為評估模式 ==========
    # 提示: model.eval()
    model.eval()


    # ========== TODO 3: 建立假輸入 (1, 3, 224, 224) ==========
    # 提示: dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)


    # ========== TODO 4: 匯出 ONNX 模型 ==========
    onnx_path = "simple_cnn.onnx"
    """
    請使用 torch.onnx.export 匯出模型
    提示:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=['input'],
            output_names=['output']
        )
    """
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=['input'],
        output_names=['output']
    )


    print(f"\n模型已匯出至: {onnx_path}")
    print("\n下一步:")
    print("1. 編譯 TensorRT 引擎:")
    print("   trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine")
    print("\n2. 執行推論與效能分析:")
    print("   trtexec --loadEngine=simple_cnn_fp32.engine --dumpProfile")


if __name__ == "__main__":
    main()
