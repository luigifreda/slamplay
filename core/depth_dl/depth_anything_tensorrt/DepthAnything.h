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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Depth_estimation structure
 */
struct DepthEstimation {
    int x;
    int y;
    int label;

    DepthEstimation() {
        x = 0;
        y = 0;
        label = -1;
    }
};

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat &img, int w, int h);

class DepthAnything {
   public:
    DepthAnything(std::string model_path, nvinfer1::ILogger &logger);
    cv::Mat predict(const cv::Mat &image);
    ~DepthAnything();

   private:
    int input_w = 518;
    int input_h = 518;
    float mean[3] = {123.675, 116.28, 103.53};  // [0.485, 0.456, 0.406] * 255
    float std[3] = {58.395, 57.12, 57.375};     // [0.229, 0.224, 0.225] * 255

    std::vector<int> offset;

    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    nvinfer1::INetworkDefinition *network;

    void *buffer[2];
    float *depth_data;
    cudaStream_t stream;

    std::vector<float> preprocess(cv::Mat &image);
    std::vector<DepthEstimation> postprocess(std::vector<int> mask, int img_w, int img_h);
    void build(std::string onnxPath, nvinfer1::ILogger &logger);
    bool saveEngine(const std::string &filename);
};
