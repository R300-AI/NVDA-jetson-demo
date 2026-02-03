這份作業的核心在於學習 TensorRT 的訓練後量化 (PTQ, Post-Training Quantization) 流程。

我們將透過 NVIDIA 的工具 Polygraphy 來完成這件事。相比直接寫 C++/Python API，Polygraphy 提供了更方便的指令列工具來處理繁雜的校正 (Calibration) 流程。

以下是針對您題目的完整實作指南與 TA 解析：

準備工作：確認環境
確保您的 Jetson Orin 上已經安裝 Polygraphy。通常在安裝 TensorRT 時會包含，若沒有請安裝：

Bash
pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
假設您已經有 Practice 2 產生的 yolov8n.onnx 模型。如果沒有，請先用以下指令快速產生：

Bash
# 產生 yolov8n.onnx (若已有則跳過)
yolo export model=yolov8n.pt format=onnx opset=13
第一步：撰寫 Data Loader 腳本 (data_loader.py)
量化需要「校正資料 (Calibration Data)」來統計 Tensor 的數值分佈，以決定最佳的縮放因子 (Scale Factor)。Polygraphy 需要一個 Python 腳本來「餵」資料給它。

請建立一個檔案 data_loader.py，內容如下：

Python
import numpy as np

# 定義生成資料的函式，名稱通常用 load_data，Polygraphy 會自動呼叫
def load_data():
    # 設定校正的批次大小 (Batch Size) 和數量
    # 這裡示範產生 10 個 Batch，每個 Batch 1 張圖
    n_batches = 10
    batch_size = 1
    
    # YOLOv8 預設輸入通常是 "images"，形狀為 (1, 3, 640, 640)
    # 建議先用 polygraphy inspect model yolov8n.onnx 確認輸入節點名稱
    input_name = "images" 
    shape = (batch_size, 3, 640, 640)

    for _ in range(n_batches):
        # 題目提示使用隨機數據作為示範
        # 產生 0~1 之間的 float32 亂數
        data = np.random.uniform(0, 1, shape).astype(np.float32)
        
        # 必須回傳一個字典 (Dictionary)，Key 是輸入節點名稱，Value 是資料
        yield {input_name: data}
TA 提示： 在真實專案中，這裡不能用 np.random，必須讀取真實的訓練圖片並做前處理 (Resize, Normalize)，否則量化出來的模型精度會完全爛掉（因為它統計到了錯誤的數值分佈）。但作業僅是練習「流程」，所以用亂數是可以的。

第二步：執行 Polygraphy 進行 INT8 量化
使用 polygraphy convert 指令，將 ONNX 轉為 TensorRT Engine，並開啟 INT8 模式。

請在終端機執行：

Bash
polygraphy convert yolov8n.onnx \
    --int8 \
    --data-loader-script data_loader.py \
    --calibration-cache yolov8.cache \
    -o yolov8n_int8.engine
參數詳解：

--int8: 啟用 INT8 量化模式。

--data-loader-script data_loader.py: 指定剛剛寫的 Python 腳本，用來餵校正資料。

--calibration-cache yolov8.cache: 將校正結果存成 cache 檔。下次若要重新 build engine 且網路結構沒變，就可以直接讀這個檔，不用重跑校正（節省時間）。

-o yolov8n_int8.engine: 輸出編譯好的 INT8 引擎。

第三步：作業練習 - 效能分析與比較
題目要求使用 --dumpProfile 與 --dumpLayerInfo 來觀察。這通常是搭配 trtexec 工具來進行深入分析。

1. 觀察 INT8 引擎效能
執行以下指令來測試剛剛生成的 INT8 引擎：

Bash
# 測試 INT8 Engine 並輸出詳細層資訊
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=yolov8n_int8.engine \
    --dumpProfile \
    --dumpLayerInfo \
    --avgRuns=100 > int8_log.txt
2. 比較 FP16 (Practice 2) 與 INT8
假設您之前 Practice 2 產生的 FP16 引擎叫 yolov8n_fp16.engine（若沒有，請用 trtexec --onnx=yolov8n.onnx --fp16 --saveEngine=yolov8n_fp16.engine 快速生成）。

同樣跑一次測試：

Bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=yolov8n_fp16.engine \
    --avgRuns=100 > fp16_log.txt
