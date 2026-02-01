"""
Practice 5: ResNet18 的 QAT with CIFAR-10

題目說明:
1. 使用 torchvision.datasets.CIFAR10 載入 CIFAR-10 訓練集與測試集
2. 建立 ResNet18 模型，修改最後一層 fc 輸出大小為 10
3. 在模型中插入 QuantStub 與 DeQuantStub，設定 qconfig，執行 prepare_qat
4. 在 CIFAR-10 訓練集上進行 QAT 訓練
5. 完成 QAT 訓練後，將模型轉換為量化版本並匯出成 ONNX 格式

執行方式:
    python3 qat_resnet18_cifar10.py

編譯 TensorRT 引擎:
    trtexec --onnx=resnet18_qat_cifar10.onnx --saveEngine=resnet18_qat.engine \\
            --int8 --shapes=input:1x3x32x32 \\
            --dumpProfile --dumpLayerInfo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

class QuantizedResNet18(nn.Module):
    """包裝 ResNet18 並加入量化 Stub"""
    def __init__(self, num_classes=10):
        super(QuantizedResNet18, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 載入 ResNet18 並修改最後一層
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def main():
    print("=" * 60)
    print("Practice 5: ResNet18 QAT with CIFAR-10")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    # ========== TODO 1: 載入 CIFAR-10 資料集 ==========
    """
    請載入 CIFAR-10 訓練集與測試集
    提示:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                   shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                  shuffle=False, num_workers=2)
    """
    trainloader = None  # 請修改此行
    testloader = None   # 請修改此行


    # ========== TODO 2: 建立量化模型並設定 qconfig ==========
    """
    請建立量化模型並設定 qconfig
    提示:
        model = QuantizedResNet18(num_classes=10)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = prepare_qat(model.train())
    """
    model = None  # 請修改此行


    # ========== TODO 3: 訓練模型 (QAT) ==========
    """
    請訓練模型 (建議 5-10 個 epoch)
    提示:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
                    running_loss = 0.0
    """


    # ========== TODO 4: 轉換為量化模型並匯出 ONNX ==========
    """
    請轉換為量化模型並匯出 ONNX
    提示:
        model.eval()
        model_quantized = convert(model)
        
        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(
            model_quantized,
            dummy_input,
            "resnet18_qat_cifar10.onnx",
            opset_version=13,
            input_names=['input'],
            output_names=['output']
        )
    """


    print("\n下一步:")
    print("1. 編譯 TensorRT 引擎:")
    print("   trtexec --onnx=resnet18_qat_cifar10.onnx --saveEngine=resnet18_qat.engine \\")
    print("           --int8 --shapes=input:1x3x32x32 \\")
    print("           --dumpProfile --dumpLayerInfo")
    print("\n2. 比較 FP32、PTQ、QAT 三種版本的效能與準確率")

if __name__ == "__main__":
    main()
