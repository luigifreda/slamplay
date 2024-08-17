#pragma once

#include "io/io_conversions.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

// #include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace slamplay {

uint32_t getElementSize(const nvinfer1::DataType &t) noexcept {
    switch (t)
    {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
            return 1;
    }
    return 0;
}

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
   public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size), mCapacity(size), mType(type) {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer &&buf)
        : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer) {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer &operator=(GenericBuffer &&buf) {
        if (this != &buf)
        {
            if (mBuffer) freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void *data() {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void *data() const {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const {
        return this->size() * getElementSize(mType);
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize) {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            if (mBuffer) freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    //!
    //! \brief Overload of resize that accepts Dims
    //!
    void resize(const nvinfer1::Dims &dims) {
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        return this->resize(vol);
    }

    //!
    //! \brief copy data from host to device
    //!
    int host2device(void *data, bool async, const cudaStream_t &stream = 0) {
        int ret = 0;
        if (async)
            ret = cudaMemcpyAsync(mBuffer, data, nbBytes(), cudaMemcpyHostToDevice, stream);
        else
            ret = cudaMemcpy(mBuffer, data, nbBytes(), cudaMemcpyHostToDevice);
        return ret;
    }

    //!
    //! \brief copy data from device to host
    //!
    int device2host(void *data, bool async, const cudaStream_t &stream = 0) {
        int ret = 0;
        if (async)
            ret = cudaMemcpyAsync(data, mBuffer, nbBytes(), cudaMemcpyDeviceToHost, stream);
        else
            ret = cudaMemcpy(data, mBuffer, nbBytes(), cudaMemcpyDeviceToHost);
        return ret;
    }

    //!
    //! \brief copy data from device to host
    //!
    int device2device(void *data, bool async, const cudaStream_t &stream = 0) {
        int ret = 0;
        if (async)
            ret = cudaMemcpyAsync(data, mBuffer, nbBytes(), cudaMemcpyDeviceToDevice, stream);
        else
            ret = cudaMemcpy(data, mBuffer, nbBytes(), cudaMemcpyDeviceToDevice);
        return ret;
    }

    nvinfer1::DataType getDataType() {
        return mType;
    }

    ~GenericBuffer() {
        if (mBuffer) freeFn(mBuffer);
    }

   private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void *mBuffer = nullptr;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator {
   public:
    bool operator()(void **ptr, size_t size) const {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree {
   public:
    void operator()(void *ptr) const {
        if (ptr) cudaFree(ptr);
    }
};
using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;

}  // namespace slamplay
