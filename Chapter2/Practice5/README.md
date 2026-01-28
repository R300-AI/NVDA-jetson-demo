# 透過指標減少Reshape的資料搬移

### 題目說明
1. 利用 `Eigen::MatrixXf::Random(1024, 768)` 在 CPU 建立一個隨機浮點數矩陣 A
2. 使用 `reshaped()` 將其轉換為 (768×1024) 向量
3. 利用 `wp.from_ptr` 將該矩陣的指標映射至 GPU
4. 使用 `wp.launch()` 計算該向量的內積 A·A

### 作業練習
* 在 CPU 上印出 reshape 前後的首元素位址，確認是否相同（位址相同 → 沒有進行搬移）
* 使用 `nsys profile` 觀察 **Memory Operations**，確認無 Host-Device 資料傳輸
* 觀察報告中是否有 `cudaMemcpy` 呼叫，驗證 Zero-Copy 是否生效
