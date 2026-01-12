cmake_minimum_required(VERSION 3.10)
project(JetsonAdvancedCUDA)

# 尋找必要的套件
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# --- [新增這行] 強制加入 CUDA 函式庫路徑 ---
link_directories(/usr/local/cuda/lib64)
# ----------------------------------------

# 設定編譯參數 (針對 Jetson Orin 架構 sm_87 優化)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_87 -O3")

include_directories(${EIGEN3_INCLUDE_DIR})

# --- 下面維持不變 ---
cuda_add_executable(lab1 lab1_cpu_vs_gpu.cu)

cuda_add_executable(lab2 lab2_divergence.cu)

cuda_add_executable(lab3 lab3_cublas.cu)
target_link_libraries(lab3 cublas)  # 現在它會找得到 cublas 了

cuda_add_executable(lab4 lab4_attention.cu)
target_link_libraries(lab4 cublas)

cuda_add_executable(lab5 lab5_reshape.cu)

cuda_add_executable(lab6 lab6_norm.cu)

cuda_add_executable(lab7_8 lab7_8_neuron.cu)
target_link_libraries(lab7_8 cublas)