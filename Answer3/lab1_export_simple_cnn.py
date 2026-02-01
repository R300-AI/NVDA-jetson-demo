"""Lab 1 解答: TensorRT 基本部署流程"""

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


model = SimpleCNN(num_classes=1000)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "simple_cnn.onnx", opset_version=17, input_names=['input'], output_names=['output'])
