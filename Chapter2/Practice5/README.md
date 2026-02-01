# 透過指標減少 Reshape 的資料搬移

### 題目說明
1. 利用 `cudaMallocManaged` 建立一個隨機浮點數矩陣 A (1024 x 768)
2. 將其視為一維向量（不複製資料，使用同一個指標）
3. 利用指標映射至 GPU
4. 實作 GPU Kernel 計算該向量的內積 A·A

### 作業練習
* 印出 reshape 前後的首元素位址，確認是否相同（位址相同 → 沒有進行搬移）
* 通過 nsys 追蹤 cuda 效能：
  * 在 CUDA API Summary 中確認沒有 cudaMemcpy 或 cudaMemcpyAsync 呼叫
  * 在 Timeline View 中觀察 dot_product_kernel 的執行區間，確認GPU直接存取資料
