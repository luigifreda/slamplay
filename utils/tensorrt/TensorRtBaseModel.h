#pragma once

#include "tensorrt/DeviceBuffer.h"
#include "tensorrt/tensorrt_utils.h"

namespace slamplay {

class TensorRtBaseModel {
   public:
    cudaStream_t stream;
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    std::vector<void*> mDeviceBindings;
    std::map<std::string, std::unique_ptr<slamplay::DeviceBuffer>> mInOut;
    std::vector<std::string> mInputsName, mOutputsName;

   public:
    TensorRtBaseModel(std::string modelFile);
    ~TensorRtBaseModel();
    void readEngineFile(std::string modelFile);
};

TensorRtBaseModel::TensorRtBaseModel(std::string modelFile) {
    readEngineFile(modelFile);
    context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    checkCudaStatus(cudaStreamCreate(&stream));

    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        auto dims = engine->getBindingDimensions(i);
        auto tensor_name = engine->getBindingName(i);
        bool isInput = engine->bindingIsInput(i);
        if (isInput)
            mInputsName.emplace_back(tensor_name);
        else
            mOutputsName.emplace_back(tensor_name);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        dims2str(dims);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        index2srt(type);
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<slamplay::DeviceBuffer> device_buffer{new slamplay::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
}

TensorRtBaseModel::~TensorRtBaseModel() {
    checkCudaStatus(cudaStreamDestroy(stream));
    if (context)
    {
        context->destroy();
    }
    if (engine)
    {
        engine->destroy();
    }
}

void TensorRtBaseModel::readEngineFile(std::string modelFile) {
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);

    int fsize;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if (engineFile)
        std::cout << "all characters read successfully." << std::endl;
    else
        std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
    engineFile.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

}  // namespace slamplay