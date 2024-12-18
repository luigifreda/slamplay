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
#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include "features_dl/hfnet/HFNetBaseModel.h"

namespace hfnet {

class HFNetBaseModel;

class HFextractor {
   public:
    HFextractor(int nfeatures, float threshold, HFNetBaseModel *pModels);

    HFextractor(int nfeatures, float threshold, float scaleFactor,
                int nlevels, const std::vector<HFNetBaseModel *> &vpModels);

    ~HFextractor() {}

    // Compute the features and descriptors on an image.
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint> &_keypoints,
                   cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors);

    int inline GetLevels(void) {
        return nlevels;
    }

    float inline GetScaleFactor(void) {
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors(void) {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(void) {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(void) {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(void) {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<int> mnFeaturesPerLevel;

    int nfeatures;
    float threshold;

    std::vector<HFNetBaseModel *> mvpModels;

   protected:
    double scaleFactor;
    int nlevels;
    bool bUseOctTree;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<int> umax;

    void ComputePyramid(const cv::Mat &image);

    int ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);

    int ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);

    int ExtractMultiLayersParallel(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                                   cv::Mat &localDescriptors, cv::Mat &globalDescriptors);
};

}  // namespace hfnet

#endif
