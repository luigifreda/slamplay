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
#ifndef HFNETTFMODELV2_H
#define HFNETTFMODELV2_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "features_dl/hfnet/HFNetBaseModel.h"

#ifdef USE_TENSORFLOW
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session_options.h"
#endif  // USE_TENSORFLOW

namespace hfnet {

#ifdef USE_TENSORFLOW

class HFNetTFModelV2 : public HFNetBaseModel {
   public:
    HFNetTFModelV2(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape);
    virtual ~HFNetTFModelV2(void) = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override;

    bool IsValid(void) override { return mbVaild; }

    ModelType Type(void) override { return kHFNetTFModel; }

    std::shared_ptr<tensorflow::Session> mSession;
    tensorflow::GraphDef mGraph;

   protected:
    bool LoadHFNetTFModel(const std::string &strModelDir);

    bool Run(std::vector<tensorflow::Tensor> &vNetResults);

    void GetLocalFeaturesFromTensor(const tensorflow::Tensor &tScoreDense, const tensorflow::Tensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, tensorflow::Tensor *tensor);

    void Tensor2Mat(tensorflow::Tensor *tensor, cv::Mat &mat);

    void ResamplerTF(const tensorflow::Tensor &data, const tensorflow::Tensor &warp, cv::Mat &output);

    ModelDetectionMode mMode;
    std::string mstrTFModelPath;
    bool mbVaild = false;
    std::vector<tensorflow::Tensor> mvNetResults;
    std::vector<std::pair<std::string, tensorflow::Tensor>> mvInputTensors;
    std::vector<std::string> mvOutputTensorNames;
    tensorflow::TensorShape mInputShape;
};

#else  // USE_TENSORFLOW

class HFNetTFModelV2 : public HFNetBaseModel {
   public:
    HFNetTFModelV2(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape) {
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

#endif  // HFNETTFMODELV2_H