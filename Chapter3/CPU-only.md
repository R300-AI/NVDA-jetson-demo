1️⃣ 系統更新與基礎套件
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
pip3 install --upgrade pip

2️⃣ 安裝 CPU-only PyTorch
# 卸載舊版（如果已經裝過）
pip3 uninstall -y torch torchvision torchaudio

# 安裝官方 CPU-only 版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

3️⃣ 安裝其他必要套件
pip3 install numpy<2 pillow opencv-python onnx onnxruntime


numpy<2：避免某些舊範例與 PyTorch 版本不兼容

onnx：PyTorch 匯出 ONNX 必須

onnxruntime：CPU-only 推論

pillow / opencv-python：影像處理用

4️⃣ 匯出 ONNX 模型
python3 lab1_export_simple_cnn.py


輸出檔案：simple_cnn.onnx

⚠️ 確保模型使用 CPU：

model.to("cpu")
