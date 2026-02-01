最可能的原因：
原因	解決方案
nsys 版本不匹配 JetPack	sudo apt install nsight-systems-2024.5 重裝官方版本
GPU 時間戳同步問題	加 --gpu-metrics-device=none 參數
權限不足	使用 sudo 執行
系統時鐘問題	sudo timedatectl set-ntp true 後重啟
我已建立 TROUBLESHOOTING.md 文件，包含完整的診斷與修復步驟。請先試試 方案 B（不用 --capture-


# Nsight Systems `TimeConversion.cpp` 錯誤排除指南

## 錯誤訊息

```
FATAL ERROR: /dvs/p4/build/sw/devtools/Agora/Rel/QuadD_Main/QuadD/Target/Daemon/TimeConversion.cpp(504): 
Throw in function ConvertGpuTicksToSyncNs
Dynamic exception type: boost::wrapexcept<QuadDCommon::InternalErrorException>
```

## 問題原因

這是 **Nsight Systems (nsys) 內部錯誤**，與您的 CUDA 程式碼無關。

問題通常出在：
1. **nsys 版本與 JetPack 不匹配**
2. **GPU 時間戳同步失敗**
3. **系統時間或時鐘源問題**

---

## 診斷步驟

### 1. 檢查 nsys 版本

```bash
nsys --version
```

**JetPack 6.2 官方版本應該是 `2024.5.x`**

如果顯示其他版本（如 2026.1.1），表示 nsys 被升級過，可能與 JetPack 不相容。

### 2. 檢查 JetPack 版本

```bash
cat /etc/nv_tegra_release
# 或
dpkg -l | grep nvidia-jetpack
```

### 3. 檢查 CUDA 驅動版本

```bash
cat /usr/local/cuda/version.txt
nvidia-smi  # 可能無法在 Jetson 上使用
nvcc --version
```

---

## 解決方案

### 方案 A：重新安裝 JetPack 官方 nsys 版本（推薦）

```bash
# 查看目前安裝的 nsys
dpkg -l | grep nsight-systems

# 移除非官方版本
sudo apt remove nsight-systems-*

# 重新安裝 JetPack 官方版本
sudo apt update
sudo apt install nsight-systems-2024.5
```

### 方案 B：不使用 `--capture-range=cudaProfilerApi`

如果您之前不用 profiler API 就能成功，嘗試：

```bash
# 使用基本追蹤模式（不指定 capture-range）
nsys profile --trace=cuda -o output_trace ./your_program

# 或使用 timeline 模式
nsys profile -t cuda,nvtx -o output_trace ./your_program
```

### 方案 C：使用 `--gpu-metrics-device=none` 跳過 GPU 時間戳

```bash
nsys profile --trace=cuda --gpu-metrics-device=none -o output_trace ./your_program
```

### 方案 D：確保系統時間同步

```bash
# 檢查系統時間
timedatectl status

# 同步 NTP 時間
sudo timedatectl set-ntp true

# 重新啟動後再試
sudo reboot
```

### 方案 E：使用 sudo 執行 nsys

在 Tegra 平台上，CUDA trace 需要 root 權限：

```bash
sudo nsys profile --trace=cuda -o output_trace ./your_program
```

---

## 移除 cudaProfilerApi 的程式碼版本

如果上述方案無效，可以暫時移除 profiler API，使用最基本的模式：

**編譯指令：**
```bash
nvcc program.cu -o program -O2 -arch=sm_87
```

**執行指令：**
```bash
# 不使用 capture-range
nsys profile --trace=cuda -o program_trace ./program
```

---

## 確認修復

成功後應該看到：

```
Generating '/tmp/nsys-report-xxx.qdstrm'
[1/1] [========================100%] output_trace.nsys-rep
Generated:
    /path/to/output_trace.nsys-rep
```

而不是 `FATAL ERROR`。

---

## 參考資料

- [Nsight Systems Release Notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
- [JetPack 6.2 Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html)
- [NVIDIA Developer Forums - nsys issues](https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/116)

---

## 聯絡支援

如果所有方案都無法解決，建議到 NVIDIA Developer Forums 回報此 bug：
https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/116

請提供：
1. `nsys --version` 輸出
2. `cat /etc/nv_tegra_release` 輸出
3. 完整錯誤訊息
4. 最小可重現範例程式
