// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#pragma once

#include "tensorrt/GenericBuffer.h"
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
        slamplay::dims2str(dims);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        slamplay::index2srt(type);
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
#if NV_TENSORRT_VERSION_CODE < 100000L
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
#else
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
#endif
}

}  // namespace slamplay