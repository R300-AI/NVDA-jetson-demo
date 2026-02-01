"""
Practice 1: TensorRT 基本部署流程

執行方式:
    python3 export_simple_cnn.py
    trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine
    trtexec --loadEngine=simple_cnn_fp32.engine --dumpProfile --dumpLayerInfo
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """簡單的 CNN 圖像分類器
    
    輸入: (1, 3, 224, 224) - batch=1, RGB 3通道, 224x224 圖像
    輸出: (1, 10) - 10 個分類
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        # ========== TODO 1: 定義網路層 ==========
        # 提示: 使用 nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Flatten, nn.Linear
        # 參考 Chapter3 README「匯出自訂 ONNX 模型」的範例架構
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128 * 28 * 28, num_classes)
    
    def forward(self, x):
        # ========== TODO 2: 實作前向傳播 ==========
        # 提示: x 經過 self.features 後，再經過 self.classifier
        
        x = self.features(x)
        return self.classifier(x)


# ========== 匯出 ONNX ==========
print("【Practice 1】匯出 SimpleCNN 為 ONNX 格式")

model = SimpleCNN(num_classes=10)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
output_path = "simple_cnn.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    output_path, 
    opset_version=17, 
    input_names=['input'], 
    output_names=['output']
)

print(f"已匯出: {output_path}")
print(f"輸入名稱: input, 形狀: {tuple(dummy_input.shape)}")
print("下一步:")
print("  trtexec --onnx=simple_cnn.onnx --saveEngine=simple_cnn_fp32.engine")
print("  trtexec --loadEngine=simple_cnn_fp32.engine --dumpProfile --dumpLayerInfo")
