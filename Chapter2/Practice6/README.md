# Normalization 的記憶體層級差異

### 題目說明
1. 利用 Eigen::MatrixXf::Random(512, 768) 在 CPU 建立一個隨機浮點數矩陣 A
2. 使用 wp.reduce_sum() 計算 Mean 與 Std，並以 wp.launch() 計算(A−Mean)/Std
3. 利用 std::chrono 記錄整體執行時間

### 作業練習
* 使用 tegrastats 監測 GPU 的功耗 (POM_5V_GPU)、使用率 (GR3D_FREQ) 與記憶體 (RAM)
* 將矩陣逐步放大（512 → 2048 → 4096 → 8192），觀察效能是否呈縣性增長?
