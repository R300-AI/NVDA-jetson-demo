# Jetson Orin 程式實作專案 (1)

這個學習資源旨在讓你快速掌握 **Jetson Orin** 平台的開發流程。你將學會如何設定環境、編譯並執行以 `Eigen` 和 `OpenBLAS` 為基礎的矩陣運算程式，並透過 `tegrastats` 工具即時觀察 CPU、GPU 及記憶體的使用狀況。

## 準備環境

本教材以 Jetson Orin + JetPack 6.2 為例，如果你的系統尚未安裝 **Ubuntu with JetPack**，請先依照 [官方教學](https://developer.nvidia.com/embedded/jetpack-sdk-62)完成安裝。

1. 安裝開發工具與數學運算函式庫
  ```bash
  sudo apt update
  sudo apt install -y build-essential libeigen3-dev libopenblas-dev
  ```
2. 確認函式庫的安裝路徑，方便後續編譯程式碼
  ```bash
  # 查找 Eigen 標頭檔路徑（通常為 /usr/include/eigen3）
  find /usr/include -type d -name "eigen3"

  # 查找 OpenBLAS 函式庫路徑（通常為 /usr/lib/aarch64-linux-gnu）
  find /usr/lib -name "libopenblas.so*"
  ```

## 編譯與執行

1. 使用你查詢到的 Eigen 標頭檔路徑與 OpenBLAS 函式庫路徑，執行編譯指令
```bash
g++ <source_file>.cpp -o <output_binary> -I <eigen_include_path> -L <openblas_lib_path> -lopenblas
```
* `<source_file>.cpp`：你的程式碼檔案
* `<output_binary>`：編譯後的執行檔名稱
* `<eigen_include_path>`：Eigen 標頭檔路徑
* `<openblas_lib_path>`：OpenBLAS 函式庫路徑

2. 執行程式：
```bash
./<output_binary>
```
3. 開啟效能監測（可於另一個命令列執行）：
```bash
sudo tegrastats
```

## 進階練習題

1. 請使用 `Eigen`  實作 100x100 矩陣的「加法」與「乘法」運算（C = A + B, C = A × B），並用 `tegrastats` 測量 Jetson Orin 的 CPU、記憶體使用量。

2. 請使用 `OpenBLAS`  實作 100x100 矩陣的「加法」與「乘法」運算，並測量 Jetson Orin 的 CPU、記憶體使用量。

3. 延續第二題，，請加入 `openblas_set_num_threads(2)`;，分別比較單執行緒與多執行緒的執行時間插一，並觀察 CPU 的使用量。

## Eigen, OpenBLAS範例解答
### eigen_3x3.cpp
```cpp
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

int main() {
    Matrix3d A, B, C;
    A << 1,2,3,4,5,6,7,8,9;
    B << 9,8,7,6,5,4,3,2,1;
    C = A + B;
    cout << "A + B =\n" << C << endl;
    C = A * B;
    cout << "A * B =\n" << C << endl;
    return 0;
}
```
### openblas_1000.cpp
```cpp
#include <iostream>
#include <cblas.h>
#include <vector>
using namespace std;

int main() {
    int N = 1000;
    vector<double> A(N*N, 1.0), B(N*N, 2.0), C(N*N, 0.0);

    // 矩陣加法
    for(int i=0; i<N*N; ++i) C[i] = A[i] + B[i];

    // 矩陣乘法
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);

    cout << "OpenBLAS 運算完成" << endl;
    return 0;
}
```
如需多執行緒，請在 main 開頭加上：
```cpp
#include <openblas_config.h>
openblas_set_num_threads(2);
```
