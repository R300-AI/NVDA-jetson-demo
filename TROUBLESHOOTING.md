# Nsight Systems `TimeConversion.cpp` 錯誤排除指南

## 錯誤訊息

```
FATAL ERROR: /dvs/p4/build/sw/devtools/Agora/Rel/QuadD_Main/QuadD/Target/Daemon/TimeConversion.cpp(504): 
Throw in function ConvertGpuTicksToSyncNs
Dynamic exception type: boost::wrapexcept<QuadDCommon::InternalErrorException>
```

## 系統資訊（診斷結果）

```bash
# Jetson Linux 版本
cat /etc/nv_tegra_release
# R36 (release), REVISION: 4.7 → 這是 JetPack 6.2

# CUDA 版本
nvcc --version
# Cuda compilation tools, release 12.6, V12.6.68
```

---

## 問題原因

這是 **Nsight Systems (nsys) 內部錯誤**，與 CUDA 程式碼無關。

可能原因：
1. **nsys 版本與 JetPack/L4T 不相容**
2. **GPU 時間戳同步失敗**
3. **`--capture-range=cudaProfilerApi` 在某些版本有 bug**

---

## 解決方案

### ⚠️ 前置作業：修復 apt 鏡像問題

如果 `sudo apt update` 出現以下錯誤：
```
E: 無法取得 http://tw.archive.ubuntu.com/ubuntu/dists/jammy-updates/main/binary-arm64/Packages.xz
檔案包含非預期的大小...進行鏡像同步？
```

**這是台灣 Ubuntu 鏡像伺服器同步問題，請執行以下步驟切換到主伺服器：**

```bash
# 步驟 1：備份原始 sources.list
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup

# 步驟 2：將台灣鏡像替換為主伺服器
sudo sed -i 's/tw.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list

# 步驟 3：清除 apt 快取
sudo rm -rf /var/lib/apt/lists/*

# 步驟 4：重新更新
sudo apt update
```

**或者，使用其他亞洲鏡像（可能更快）：**

```bash
# 使用日本鏡像
sudo sed -i 's/tw.archive.ubuntu.com/jp.archive.ubuntu.com/g' /etc/apt/sources.list

# 或使用新加坡鏡像
sudo sed -i 's/tw.archive.ubuntu.com/sg.archive.ubuntu.com/g' /etc/apt/sources.list

# 然後清除快取並更新
sudo rm -rf /var/lib/apt/lists/*
sudo apt update
```

---

### 方案 A：重新安裝 JetPack 官方 nsys 版本（推薦）

**修復 apt 後**執行：

```bash
# 1. 查看目前安裝的 nsys 版本
nsys --version
dpkg -l | grep nsight-systems

# 2. 移除現有版本
sudo apt remove nsight-systems-*

# 3. 重新安裝（JetPack 6.2 官方版本）
sudo apt update
sudo apt install nsight-systems-2024.5

# 4. 確認版本
nsys --version
```
---

### 方案 B：不使用 `--capture-range=cudaProfilerApi`（快速測試）

如果之前不用 profiler API 就能成功，請嘗試：

```bash
# 直接執行，不指定 capture-range
nsys profile --trace=cuda -o output_trace ./your_program
```

---

### 方案 C：使用 `--gpu-metrics-device=none` 跳過 GPU 時間戳

```bash
nsys profile --trace=cuda --gpu-metrics-device=none -o output_trace ./your_program
```

---

### 方案 D：使用 sudo 執行 nsys

在 Tegra 平台上，CUDA trace 可能需要 root 權限：

```bash
sudo nsys profile --trace=cuda -o output_trace ./your_program
```

---

### 方案 E：從 NVIDIA 官網手動下載 nsys

如果 apt 安裝不成功，可以手動下載：

1. 前往 https://developer.nvidia.com/nsight-systems
2. 下載 **Linux ARM64 (AArch64)** 版本
3. 選擇與 JetPack 6.2 相容的版本（2024.5.x 或 2024.6.x）

```bash
# 解壓縮後直接使用
chmod +x nsight-systems-*/bin/nsys
./nsight-systems-*/bin/nsys --version
./nsight-systems-*/bin/nsys profile --trace=cuda -o output_trace ./your_program
```

---

## 移除 cudaProfilerApi 的程式碼

如果需要暫時移除 profiler API：

**修改前：**
```cpp
#include <cuda_profiler_api.h>
// ...
cudaProfilerStart();
kernel<<<...>>>();
cudaDeviceSynchronize();
cudaProfilerStop();
```

**修改後：**
```cpp
// 移除 #include <cuda_profiler_api.h>
// ...
// 移除 cudaProfilerStart();
kernel<<<...>>>();
cudaDeviceSynchronize();
// 移除 cudaProfilerStop();
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

---

## 建議測試順序

| 順序 | 方案 | 說明 |
|------|------|------|
| 1 | B | 不用 `--capture-range=cudaProfilerApi` |
| 2 | C | 加 `--gpu-metrics-device=none` |
| 3 | D | 用 `sudo` 執行 |
| 4 | A | 修復 apt 後重新安裝 nsys |
| 5 | E | 手動下載 nsys |

---

## 參考資料

- [Nsight Systems Release Notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
- [JetPack 6.2 Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html)
- [NVIDIA Developer Forums - nsys issues](https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/116)

---

## 如果所有方案都無效

請到 NVIDIA Developer Forums 回報：
https://forums.developer.nvidia.com/c/developer-tools/nsight-systems/116

提供以下資訊：
1. `nsys --version`
2. `cat /etc/nv_tegra_release`
3. 完整錯誤訊息
4. 使用的 nsys 指令
