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
//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_GLUE_H_
#define SUPER_GLUE_H_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include <tensorrtbuffers/buffers.h>
#include "SuperPointGlueConfig.h"

using tensorrt_common::TensorRTUniquePtr;

class SuperGlue {
   public:
    explicit SuperGlue(const SuperGlueConfig &superglue_config);

    bool build();

    bool infer(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
               const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
               Eigen::VectorXi &indices0,
               Eigen::VectorXi &indices1,
               Eigen::VectorXd &mscores0,
               Eigen::VectorXd &mscores1);

    int matching_points(Eigen::Matrix<double, 259, Eigen::Dynamic> &features0, Eigen::Matrix<double, 259, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection = false);

    Eigen::Matrix<double, 259, Eigen::Dynamic> normalize_keypoints(
        const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height);

    void save_engine();

    bool deserialize_engine();

    void getInputOutputNames();

   private:
    SuperGlueConfig superglue_config_;
    std::vector<int> indices0_;
    std::vector<int> indices1_;
    std::vector<double> mscores0_;
    std::vector<double> mscores1_;

    nvinfer1::Dims keypoints_0_dims_{};
    nvinfer1::Dims scores_0_dims_{};
    nvinfer1::Dims descriptors_0_dims_{};
    nvinfer1::Dims keypoints_1_dims_{};
    nvinfer1::Dims scores_1_dims_{};
    nvinfer1::Dims descriptors_1_dims_{};
    nvinfer1::Dims output_scores_dims_{};

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    std::map<std::string, std::string> inOut_;  //!< Input and output mapping of the network
    std::map<std::string, nvinfer1::Dims> inOutDims_;
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime_;

    bool isVerbose_{false};

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool process_input(const tensorrt_buffers::BufferManager &buffers,
                       const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                       const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1);

    bool process_output(const tensorrt_buffers::BufferManager &buffers,
                        Eigen::VectorXi &indices0,
                        Eigen::VectorXi &indices1,
                        Eigen::VectorXd &mscores0,
                        Eigen::VectorXd &mscores1);
};

typedef std::shared_ptr<SuperGlue> SuperGluePtr;

#endif  // SUPER_GLUE_H_
