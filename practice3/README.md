# Jetson Orin 程式實作專案 (2)

這個學習資源旨在讓你體驗 **TensorRT** 與 **TensorRT Profiler** 在 Jetson Orin 上的開發流程。你將學會如何載入 ONNX 模型，選擇 GPU 或 DLA 執行推論，並比較 FP32、FP16 與 TensorRT 的定位與 **整數量化**。

## 準備環境 

確認系統已安裝 TensorRT（JetPack 已內建）。
```
# 確認 TensorRT 版本
dpkg -l | grep tensorrt

# 查找 TensorRT 標頭檔與函式庫路徑
find /usr/include -name "NvInfer.h"
find /usr/lib/aarch64-linux-gnu -name "libnvinfer.so*"
 ```

## 編譯與執行

1. 使用 TensorRT 執行編譯指令：

```bash
g++ <source_file>.cpp -o <output_binary> \
    -I <tensorrt_include_path> -L <tensorrt_lib_path> \
    -lnvinfer -lcudart
```
2. 執行程式：

```bash
./<output_binary>
```

## 進階練習題


## 範例解答
### trt_profiler.cpp
```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace nvinfer1;
using namespace std;

class MyProfiler : public IProfiler {
public:
    struct Record { string layerName; float timeMs; };
    vector<Record> records;

    void reportLayerTime(const char* layerName, float timeMs) noexcept override {
        records.push_back({layerName, timeMs});
    }

    void print() {
        cout << "==== TensorRT Profiler Report ====" << endl;
        for (auto& r : records) {
            cout << r.layerName << " : " << r.timeMs << " ms" << endl;
        }
    }
};

int main(){
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if(!parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING))){
        cerr << "Failed to parse ONNX model" << endl;
        return -1;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    MyProfiler profiler;
    context->setProfiler(&profiler);

    // 建立 input/output buffer
    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");
    size_t inputSize = 1 * 3 * 224 * 224 * sizeof(float); // 假設模型輸入為 1x3x224x224
    size_t outputSize = 1 * 1000 * sizeof(float);         // 假設模型輸出為 1000 類別

    void* d_input; void* d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 建立 bindings
    void* bindings[2];
    bindings[inputIndex] = d_input;
    bindings[outputIndex] = d_output;

    // 執行推論
    context->enqueueV2(bindings, 0, nullptr);

    // 輸出 profiler 結果
    profiler.print();

    // 清理資源
    cudaFree(d_input);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
```
