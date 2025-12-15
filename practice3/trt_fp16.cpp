#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace nvinfer1;
using namespace std;

int main(){
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    IBuilderConfig* config = builder->createBuilderConfig();

    // Step 1: 啟用 FP16 精度
    config->setFlag(BuilderFlag::kFP16);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    cout << "Engine built with FP16 precision" << endl;

    // TODO: 配置 input/output buffer
    // TODO: 執行推論，並用 profiler 觀察 FP32 vs FP16 的延遲差異

    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
