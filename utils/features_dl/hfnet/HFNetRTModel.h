#ifndef HFNETRTMODEL_H
#define HFNETRTMODEL_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "features_dl/hfnet/BaseModel.h"

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <tensorrtbuffers/buffers.h>
#endif  // USE_TENSORRT

namespace hfnet {

#ifdef USE_TENSORRT

class RTLogger : public nvinfer1::ILogger {
   public:
    RTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING) : level(severity) {}

    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const *msg) noexcept override;

    nvinfer1::ILogger::Severity level;
};

class RTTensor {
   public:
    RTTensor(void *d, nvinfer1::Dims s) : data(d), shape(s) {}
    void *data;
    nvinfer1::Dims shape;
};

class HFNetRTModel : public BaseModel {
    using BufferManager = tensorrt_buffers::BufferManager;

   public:
    HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape);
    virtual ~HFNetRTModel(void) = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override;

    bool IsValid(void) override { return mvValid; }

    ModelType Type(void) override { return kHFNetTFModel; }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;

   protected:
    bool LoadHFNetTRModel(void);

    void LoadTimingCacheFile(const std::string &strFileName, std::unique_ptr<nvinfer1::IBuilderConfig> &config, std::unique_ptr<nvinfer1::ITimingCache> &timingCache);

    void UpdateTimingCacheFile(const std::string &strFileName, std::unique_ptr<nvinfer1::IBuilderConfig> &config, std::unique_ptr<nvinfer1::ITimingCache> &timingCache);

    std::string DecideEigenFileName(const std::string &strEngineSaveDir, ModelDetectionMode mode, const nvinfer1::Dims4 inputShape);

    bool SaveEngineToFile(const std::string &strEngineSaveFile, const std::unique_ptr<nvinfer1::IHostMemory> &serializedEngine);

    bool LoadEngineFromFile(const std::string &strEngineSaveFile);

    void PrintInputAndOutputsInfo(std::unique_ptr<nvinfer1::INetworkDefinition> &network);

    bool Run(void);

    void GetLocalFeaturesFromTensor(const RTTensor &tScoreDense, const RTTensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const RTTensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, RTTensor &tensor);

    void Tensor2Mat(const RTTensor &tensor, cv::Mat &mat);

    void ResamplerRT(const RTTensor &data, const cv::Mat &warp, cv::Mat &output);

    nvinfer1::Dims4 mInputShape;
    ModelDetectionMode mMode;
    std::string mStrTRModelDir;
    std::string mStrONNXFile;
    std::string mStrCacheFile;
    bool mvValid = false;
    RTLogger mLogger;
    std::unique_ptr<BufferManager> mpBuffers;
    std::vector<RTTensor> mvInputTensors;
    std::vector<RTTensor> mvOutputTensors;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
};

#else  // USE_TENSORRT

class HFNetRTModel : public BaseModel {
   public:
    HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape) {
        std::cerr << "You must set USE_TENSORRT in CMakeLists.txt to enable tensorRT function." << std::endl;
        exit(-1);
    }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override { return false; }

    bool IsValid(void) override { return false; }

    ModelType Type(void) override { return kHFNetRTModel; }
};

#endif  // USE_TENSORRT

}  // namespace hfnet

#endif  // HFNETRTMODEL_H