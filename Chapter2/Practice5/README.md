# 透過指標減少 Reshape 的資料搬移

### 題目說明
1. 利用 `cudaMallocManaged` 建立一個隨機浮點數矩陣 A (1024 x 768)
2. 將其視為一維向量（不複製資料，使用同一個指標）
3. 利用指標映射至 GPU
4. 實作 GPU Kernel 計算該向量的內積 A·A

### 作業練習
* 印出 reshape 前後的首元素位址，確認是否相同（位址相同 → 沒有進行搬移）
* 使用 `nsys profile` 觀察 **Memory Operations**，確認無 Host-Device 資料傳輸
* 觀察報告中是否有 `cudaMemcpy` 呼叫，驗證 Zero-Copy 是否生效
