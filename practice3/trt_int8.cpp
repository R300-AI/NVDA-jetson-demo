#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace nvinfer1;
using namespace std;

// Step 1: 建立一個簡單的 Dummy Calibrator
class DummyCalibrator : public IInt8Calibrator {
public:
    int getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override { return false; }
    const void* readCalibrationCache(size_t& length) noexcept override { length = 0; return nullptr; }
    void writeCalibrationCache(const void* cache, size_t length) noexcept override {}
};

int main(){
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    IBuilderConfig* config = builder->createBuilderConfig();

    // Step 2: 啟用 INT8 量化
    config->setFlag(BuilderFlag::kINT8);
    DummyCalibrator calibrator;
    config->setInt8Calibrator(&calibrator);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    cout << "Engine built with INT8 quantization" << endl;

    // TODO: 配置 input/output buffer
    // TODO: 執行推論，並用 profiler 觀察 INT8 的效能與誤差

    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
