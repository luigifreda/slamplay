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
// This file is used to support the original HF-Net model.
// The original model needs TensorFlow1 and Resampler Op in tensorflow.contrib.
#ifndef HFNETTFMODEL_H
#define HFNETTFMODEL_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "features_dl/hfnet/HFNetBaseModel.h"

#ifdef USE_TENSORFLOW
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#endif  // USE_TENSORFLOW

namespace hfnet {

#ifdef USE_TENSORFLOW

class [[deprecated]] HFNetTFModel : public HFNetBaseModel {
   public:
    HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir);
    virtual ~HFNetTFModel(void) = default;

    void Compile(const cv::Vec4i inputSize, bool onlyDetectLocalFeatures);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                int nKeypointsNum, float threshold);

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) { return false; }

    bool IsValid(void) { return mbVaild; }

    ModelType Type(void) { return kHFNetTFModel; }

    std::shared_ptr<tensorflow::Session> mSession;
    tensorflow::GraphDef mGraph;

   private:
    bool LoadResamplerOp(const std::string &strResamplerDir);

    bool LoadHFNetTFModel(const std::string &strModelDir);

    void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor);

    bool Run(const cv::Mat &image, std::vector<tensorflow::Tensor> &vNetResults, bool onlyDetectLocalFeatures,
             int nKeypointsNum, float threshold);

    void GetLocalFeaturesFromTensor(const tensorflow::Tensor &tKeyPoints, const tensorflow::Tensor &tDescriptorsMap, const tensorflow::Tensor &tScoreDense,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors);

    void GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    std::string mstrTFModelPath;
    bool mbVaild;
    static bool mbLoadedResampler;
    std::vector<tensorflow::Tensor> mvNetResults;
};

#else  // USE_TENSORFLOW

class [[deprecated]] HFNetTFModel : public HFNetBaseModel {
   public:
    HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir) {
        std::cerr << "You must set USE_TENSORFLOW in CMakeLists.txt to enable TensorFlow function." << std::endl;
        exit(-1);
    }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override { return false; }

    bool IsValid(void) override { return false; }

    ModelType Type(void) override { return kHFNetTFModel; }
};

#endif  // USE_TENSORFLOW

}  // namespace hfnet

#endif  // HFNETTFMODEL_H