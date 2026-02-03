📝 Lab 1: SimpleCNN ONNX 匯出與 CPU-only Jetson 指南
目標

原 Lab 1 使用 TensorRT (trtexec) 在 GPU 上做推論

CPU-only Jetson 無 GPU 可用時，改用 PyTorch 或 ONNX Runtime 完成推論

保留原始 Lab 1 流程供對照

1️⃣ 系統更新與基礎套件
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
pip3 install --upgrade pip


⚠️ 注意：sudo apt-get update + pip3 install --upgrade pip 保證 Python 套件最新

2️⃣ 安裝 CPU-only 套件
# PyTorch CPU-only
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ONNX Runtime CPU (推論用)
pip3 install onnxruntime

# 其他必要套件
pip3 install numpy<2 pillow opencv-python onnx

3️⃣ 匯出 ONNX 模型
python3 lab1_export_simple_cnn.py


輸出檔案：simple_cnn.onnx

CPU-only PyTorch 可直接匯出

若原程式有 .cuda() 語句，請改成 .to("cpu")

4️⃣ （原 GPU 流程）加入 TensorRT 路徑
export PATH=$PATH:/usr/src/tensorrt/bin


⚠️ CPU-only Jetson 無 GPU，trtexec 無法使用
這裡僅供原始 Lab 1 對照參考
