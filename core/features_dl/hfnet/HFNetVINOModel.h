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
// This file is used to enable the HF-Net model running on the CPU with the OpenVINO library.
// But it is still too computationally expensive, and therefore the function is disabled.
#ifndef HFNETVINOMODEL_H
#define HFNETVINOMODEL_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "features_dl/hfnet/HFNetBaseModel.h"

#ifdef USE_OPENVINO
#include "openvino/openvino.hpp"
#endif  // USE_OPENVINO

namespace hfnet {

#ifdef USE_OPENVINO
class HFNetVINOModel : public HFNetBaseModel {
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

class HFNetVINOModel : public HFNetBaseModel {
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