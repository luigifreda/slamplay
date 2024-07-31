// This file is used to enable the HF-Net model running on the CPU with the OpenVINO library.
// But it is still too computationally expensive, and therefore the function is disabled.
#ifndef HFNETVINOMODEL_H
#define HFNETVINOMODEL_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "extractors/BaseModel.h"

#ifdef USE_OPENVINO
#include "openvino/openvino.hpp"
#endif  // USE_OPENVINO

namespace hfnet {

#ifdef USE_OPENVINO
class HFNetVINOModel : public BaseModel {
   public:
    HFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath, ModelDetectionMode mode, const cv::Vec4i inputShape);
    virtual ~HFNetVINOModel(void) = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors);

    void PrintInputAndOutputsInfo(void);

    bool IsValid(void) override { return mbVaild; }

    ModelType Type(void) override { return kHFNetVINOModel; }

    std::shared_ptr<ov::Model> mpModel;
    std::shared_ptr<ov::CompiledModel> mpExecutableNet;
    std::shared_ptr<ov::InferRequest> mInferRequest;

   protected:
    bool LoadHFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath);

    bool Run(void);

    void GetLocalFeaturesFromTensor(const ov::Tensor &tScoreDense, const ov::Tensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const ov::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, ov::Tensor *tensor);

    void Tensor2Mat(ov::Tensor *tensor, cv::Mat &mat);

    void ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output);

    static ov::Core core;

    ModelDetectionMode mMode;
    std::string mStrXmlPath;
    std::string mStrBinPath;
    bool mbVaild;
    std::vector<ov::Tensor> mvNetResults;
    ov::Tensor mInputTensor;
    std::vector<std::string> mvOutputTensorNames;
    ov::Shape mInputShape;
};

#else  // USE_OPENVINO

class HFNetVINOModel : public BaseModel {
   public:
    HFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath, ModelDetectionMode mode, const cv::Vec4i inputShape) {
        std::cerr << "You must set USE_OPENVINO in CMakeLists.txt to enable OpenVINO function." << std::endl;
        exit(-1);
    }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override { return false; }

    bool IsValid(void) override { return false; }

    ModelType Type(void) override { return kHFNetVINOModel; }
};

#endif  // USE_OPENVINO

}  // namespace hfnet

#endif  // HFNETVINOMODEL_H