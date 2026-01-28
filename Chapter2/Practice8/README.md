# Activation Functions的處理效能

### 題目說明
1. 使用 wp.from_ptr 將上一題的結果矩陣 C′ 映射為 Warp 陣列
2. 使用 wp.launch() 在 GPU 上對 C′ 進行 ReLU 操作，得到結果矩陣 𝑅
R=ReLU(C′)=max⁡(0,C+b)
3. 利用 std::chrono 記錄整體執行時間

### 作業練習
* 使用 tegrastats 觀察 GPU 功耗 (POM_5V_GPU)、利用率 (GR3D_FREQ) 與記憶體占用 (RAM)
* 與題目 7 的 神經網路運算做比較，比對兩者的能耗與負載差異。
