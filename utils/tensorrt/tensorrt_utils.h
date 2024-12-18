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
