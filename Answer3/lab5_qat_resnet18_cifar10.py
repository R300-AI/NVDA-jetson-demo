"""
Practice 5: ResNet18 的 QAT with CIFAR-10

題目說明:
1. 使用 download_cifar10() 下載並載入 CIFAR-10 訓練集與測試集
2. 使用 timm 建立 ResNet18 模型，修改最後一層 fc 輸出大小為 10
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

import os
import pickle
import tarfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
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


def load_cifar10_batch(filepath):
    """載入單一 CIFAR-10 batch"""
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    images = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(data_dict[b'labels'])
    return images, labels


def load_cifar10(data_dir):
    """載入完整 CIFAR-10 資料集"""
    # 載入訓練集 (5 個 batch)
    train_images, train_labels = [], []
    for i in range(1, 6):
        images, labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_images.append(images)
        train_labels.append(labels)
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    
    # 載入測試集
    test_images, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    
    # 標準化 (CIFAR-10 mean/std)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    
    return (train_images.astype(np.float32), train_labels,
            test_images.astype(np.float32), test_labels)


class QuantizedResNet18(nn.Module):
    """包裝 ResNet18 並加入量化 Stub"""
    def __init__(self, num_classes=10):
        super(QuantizedResNet18, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 使用 timm 載入 ResNet18 並修改最後一層
        self.model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    
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
    請下載並載入 CIFAR-10 訓練集與測試集
    提示:
        cifar_dir = download_cifar10('./data')
        train_images, train_labels, test_images, test_labels = load_cifar10(cifar_dir)
        
        trainset = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels).long())
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_labels).long())
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    """
    cifar_dir = download_cifar10('./data')
    train_images, train_labels, test_images, test_labels = load_cifar10(cifar_dir)
    
    trainset = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels).long())
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_labels).long())
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


    # ========== TODO 2: 建立量化模型並設定 qconfig ==========
    """
    請建立量化模型並設定 qconfig
    提示:
        model = QuantizedResNet18(num_classes=10)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = prepare_qat(model.train())
    """
    model = QuantizedResNet18(num_classes=10)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model.train())
    model = model.to(device)


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
    
    print("\n訓練完成!")

    # ========== 測試模型準確率 ==========
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'QAT 模型準確率: {100 * correct / total:.2f}%')


    # ========== TODO 4: 轉換為量化模型並匯出 ONNX ==========
    """
    請轉換為量化模型並匯出 ONNX
    提示:
        model.eval()
        model_cpu = model.to('cpu')
        model_quantized = convert(model_cpu)
        
        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(
            model_quantized,
            dummy_input,
            "resnet18_qat_cifar10.onnx",
            opset_version=17,
            input_names=['input'],
            output_names=['output']
        )
    """
    model.eval()
    model_cpu = model.to('cpu')
    model_quantized = convert(model_cpu)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model_quantized,
        dummy_input,
        "resnet18_qat_cifar10.onnx",
        opset_version=17,
        input_names=['input'],
        output_names=['output']
    )

    print("\n模型已匯出至: resnet18_qat_cifar10.onnx")

    print("\n下一步:")
    print("1. 編譯 TensorRT 引擎:")
    print("   trtexec --onnx=resnet18_qat_cifar10.onnx --saveEngine=resnet18_qat.engine \\")
    print("           --int8 --shapes=input:1x3x32x32 \\")
    print("           --dumpProfile --dumpLayerInfo")
    print("\n2. 比較 FP32、PTQ、QAT 三種版本的效能與準確率")

if __name__ == "__main__":
    main()
