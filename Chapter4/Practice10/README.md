# 將 PyTorch 模型部署到 Jetson Orin

## 題目說明
在 **NVIDIA Jetson Orin** 上完成以下部署流程並比較效能：

1. 以 **SmallCNN** 建立 Eager 版模型並量測推論吞吐量（samples/sec）。
2. 轉換為 **TorchScript trace**（固定結構模型適用）與 **TorchScript script**（可保留控制流）兩種格式，**儲存為 `.ts`** 檔案並**載入回測**。
3. 在 GPU 推論效能測試時，使用 **CUDA 同步**（`torch.cuda.synchronize()`）確保時間量測準確；並比較 **Eager vs. traced vs. scripted** 的吞吐量差異。

---

## (2) 作業練習

1. **完成部署三步驟並記錄效能**
    - 量測 `Eager / Traced / Scripted` 三者的 **samples/sec**，填入紀錄表。
    - 說明三者差異與可能原因（trace 常略快於 script；兩者皆可能快於 Eager，視模型與平台而定）。
2. **使用 tegrastats 觀察推論過程**
    
    執行 `sudo tegrastats`，同步觀察：
    
    - GPU 使用率/時脈、顯存占用
    - 三種模式執行期間的差異與峰值（若有）
3. **批次大小敏感度（選配）**
    - 修改 `bench_infer` 的 `batch` 參數（例如 32/64/128），重新量測三種模式吞吐量，觀察是否於較大 batch 更能發揮 TorchScript 的優勢。
4. **控制流測試（選配）**
    - 在 `SmallCNN` 之上的 head 增加 **資料相依 if/loop**，嘗試 trace 與 script：驗證 **script 可保留控制流**，而 **trace 不適合動態控制流** 的教材觀念。
5. **產物驗證**
    - 確認輸出 `smallcnn_traced.ts`、`smallcnn_scripted.ts` 可於另一支程式中 **`torch.jit.load()`** 成功載入與推論。
