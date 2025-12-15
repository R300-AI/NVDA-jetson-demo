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

    // Step 1: 指定 DLA 作為執行裝置
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(0); // 使用 DLA core 0

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* context = engine->createExecutionContext();

    cout << "Engine built on DLA core 0" << endl;

    // TODO: 配置 input/output buffer
    // TODO: 執行推論，並用 tegrastats 觀察 GPU/DLA 使用量

    context->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    config->destroy();
}
