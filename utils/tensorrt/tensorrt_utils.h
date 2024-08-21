#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <iostream>

#ifndef NV_TENSORRT_VERSION_CODE
#define NV_TENSORRT_VERSION_CODE (NV_TENSORRT_MAJOR * 10000L + NV_TENSORRT_MINOR * 100L + NV_TENSORRT_PATCH * 1L)
#endif

template <typename T>
inline void checkCudaStatus(const T& status) {
    if (status != 0)
    {
        std::cerr << "Cuda failure: " << status << std::endl;
        abort();
    }
}

template <typename T>
inline void checkCudaStatusNoAbort(const T& status) {
    if (status != 0)
    {
        std::cerr << "Cuda failure: " << status << std::endl;
    }
}

namespace slamplay {

inline void index2srt(const nvinfer1::DataType& dataType) {
    switch (dataType)
    {
        case nvinfer1::DataType::kFLOAT:
            std::cout << "nvinfer1::DataType::kFLOAT" << std::endl;
            break;
        case nvinfer1::DataType::kHALF:
            std::cout << "nvinfer1::DataType::kHALF" << std::endl;
            break;
        case nvinfer1::DataType::kINT8:
            std::cout << "nvinfer1::DataType::kINT8" << std::endl;
            break;
        case nvinfer1::DataType::kINT32:
            std::cout << "nvinfer1::DataType::kINT32" << std::endl;
            break;
        case nvinfer1::DataType::kBOOL:
            std::cout << "nvinfer1::DataType::kBOOL" << std::endl;
            break;
        case nvinfer1::DataType::kUINT8:
            std::cout << "nvinfer1::DataType::kUINT8" << std::endl;
            break;

        default:
            break;
    }
}

inline void dims2str(const nvinfer1::Dims& dims) {
    std::string o_s("[");
    for (size_t i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            o_s += ", ";
        o_s += std::to_string(dims.d[i]);
    }
    o_s += "]";
    std::cout << o_s << std::endl;
}

inline bool checkIsNegative(const nvinfer1::Dims& dims) {
    for (size_t i = 0; i < dims.nbDims; i++)
    {
        if (dims.d[i] < 0) return true;
    }
    return false;
}

class NvLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
    static NvLogger instance;
};

}  // namespace slamplay
