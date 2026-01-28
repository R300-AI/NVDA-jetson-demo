# Chapter 1 練習題解答

本資料夾包含 Chapter 1 所有練習題的正確解答，對應各 Practice 的題目要求。

## 編譯指令

```bash
# Practice 1: C++ 向量加法
g++ lab1_vector_add.cpp -o lab1 -O2

# Practice 2: Eigen 向量加法
g++ lab2_eigen_vector.cpp -o lab2 -O2 -I/usr/include/eigen3

# Practice 3: OpenBLAS 矩陣乘法
g++ lab3_openblas_gemm.cpp -o lab3 -O2 -lopenblas

# Practice 4: Eigen 矩陣乘法
g++ lab4_eigen_gemm.cpp -o lab4 -O2 -I/usr/include/eigen3
```

## 各練習對應檔案

| 練習 | 檔案 | 編譯指令 | 觀察重點 |
|------|------|----------|----------|
| P1 | `lab1_vector_add.cpp` | `g++ ... -O2` | CPU 單核使用率、NEON/SIMD 加速 |
| P2 | `lab2_eigen_vector.cpp` | `g++ ... -I/usr/include/eigen3` | Eigen vs std::vector 效能差異 |
| P3 | `lab3_openblas_gemm.cpp` | `g++ ... -lopenblas` | 多執行緒效能提升 |
| P4 | `lab4_eigen_gemm.cpp` | `g++ ... -I/usr/include/eigen3` | Eigen vs OpenBLAS 效能對比 |

## 執行範例

```bash
# 編譯 Lab 1
g++ lab1_vector_add.cpp -o lab1 -O2

# 執行
./lab1

# 使用 tegrastats 監控 (另開終端機)
sudo tegrastats
```

## 效能監測

在執行程式前，建議開啟 `tegrastats` 觀察：
1. CPU 頻率
2. VDD_CPU 功耗數值
3. 各核心使用率
