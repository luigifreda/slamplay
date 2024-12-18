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
#ifndef BASEMODEL_H
#define BASEMODEL_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace hfnet {

enum ModelType {
    kHFNetTFModel = 0,
    kHFNetRTModel,
    kHFNetVINOModel,
};
const std::string gStrModelTypeName[] = {"kHFNetTFModel", "kHFNetRTModel", "kHFNetVINOModel"};

enum ModelDetectionMode {
    kImageToLocalAndGlobal = 0,
    kImageToLocal,
    kImageToLocalAndIntermediate,
    kIntermediateToGlobal
};
const std::string gStrModelDetectionName[] = {"ImageToLocalAndGlobal", "ImageToLocal", "ImageToLocalAndIntermediate", "IntermediateToGlobal"};

class ExtractorNode {
   public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class HFNetBaseModel {
   public:
    virtual ~HFNetBaseModel(void) = default;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) = 0;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) = 0;

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) = 0;

    virtual bool IsValid(void) = 0;

    virtual ModelType Type(void) = 0;
};

class HFNetSettings;

void InitAllModels(HFNetSettings *settings);

void InitAllModels(const std::string &strModelPath, ModelType modelType, cv::Size ImSize, int nLevels, float scaleFactor);

std::vector<HFNetBaseModel *> GetModelVec(void);

HFNetBaseModel *GetGlobalModel(void);

HFNetBaseModel *InitTFModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

HFNetBaseModel *InitRTModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

HFNetBaseModel *InitVINOModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int minX,
                                            const int maxX, const int minY, const int maxY, const int N);

std::vector<cv::KeyPoint> NMS(const std::vector<cv::KeyPoint> &vToDistributeKeys, int width, int height, int radius);

void Resampler(const float *data, const float *warp, float *output,
               const int batch_size, const int data_height,
               const int data_width, const int data_channels, const int num_sampling_points);

}  // namespace hfnet

#endif