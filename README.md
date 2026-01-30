# NVDA-jetson-demo

## Jetson Orin 硬體架構與運算特性

1. Agenda
2. AI 正在改變人類的科技型態與生產模式
3. 傳統架構難以支撐 AI 運算，必須融合加速基建
4. 運用 Silicon IP 加速軟硬系統整合設計
5. 晶片模組從 0 到 1 的研發過程
6. GPU 架構演進脈絡 — 從圖形到通用 AI 的躍遷
7. Jetson 產品系列的架構差異
8. CPU 與 GPU：通用運算的加速與分工
9. 加速科學計算的考量面向，以 Vector Add 為例
10. Single Instruction Multiple Data（SIMD）
11. 實作範例（1）－透過 C++ 實作單核向量加法
12. 實作範例（2）－透過 Eigen 提升程式碼的可讀性
13. Simultaneous Multithreading（SMT）
14. 實作範例（3）－透過 OpenBLAS 實踐多核矩陣乘法
15. 實作範例（4）－Eigen 與 OpenBLAS 效能對比
16. 多核心並行的最佳策略
17. 進階練習（5）－旋轉機械的振動狀態即時評估

## NVIDIA GPU 演進與 CUDA 加速原理

1. Agenda
2. 半導體縮放達物理限制，使技術轉向異質架構
3. 低延遲與高吞吐量架構的設計對比
4. 實作範例（1）－CPU 與 GPU Warp 的效能差異
5. 傳統序列化程式於 Host‑Device 的運作機制
6. 透過 Memory Fabric 最小化資料搬移距離
7. Single Instruction Multiple Threads（SIMT）
8. SIMT 共享指令，無法同時執行不同運算操作
9. 實作範例（2）－GPU Warp Divergence 之效能分析
10. GPU 主要用於處理 General Matrix Multiply（GEMM）
11. 實作範例（3）－利用 cuBLAS 進行高效的線性代數運算
12. 以 Self‑Attention 的 GEMM 為例
13. 實作範例（4）－透過 GPU 加速 Self‑Attention
14. 階層式資料存取機制是造成高延遲的主因
15. 透過計算分塊（Tiling）減少資料回丟 DRAM
16. 以平行化 SRAM on Chip 提供更多鄰近的記憶體
17. 加速運算是 AI 實務的核心
18. 實作範例（5）－透過指標減少 Reshape 的資料搬移
19. 實作範例（6）－Normalization 的記憶體層級差異
20. 實作範例（7）－透過異質運算實踐 Artificial Neurons
21. 實作範例（8）－Activation Functions 的處理效能
22. 進階練習（9）－即時多感測訊號的狀態相似度搜尋

## NVIDIA DLA 深度學習加速器應用
1. Agenda
2. 透過物理電路實現極致的AI推論能效
3. 剖析 DLA 上的常見的硬體映射運算子
4. NVIDIA DLA在不同模型下的算力與效能表現
5. 藉由任務分流 (Offloading)釋放異質晶片的算力效能
6. 實作範例 (1)：TensorRT基本部署流程 (to GPU)
7. YOLO開源社群：一個自訂模型的發展過程
8. 從模型架構的迭代設計與運算操作方式的改變
9. ONNX 模型標準化與 YOLO 部署實務
10. 深度剖析訓練與部署框架的操作邊界
11. 神經網路的運算子結構—利用 Netron 解析操作流程
12. 實作範例 (2)：導出YOLO模型並以於FP16執行推論
13. 實作範例 (3)： DLA Operator 驗證與 Fallback 練習
14. 軟硬體共同設計最佳化實務：量化 (Quantization)
15. 模型量化的精度補償技術
16. 實作範例 (4)： 透過PyTorch實現一個基本的QAT
17. 實作範例 (5)：ResNet18 的 QAT with CIFAR-10

