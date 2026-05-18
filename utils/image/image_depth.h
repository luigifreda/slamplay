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

#include "image_error.h"

#include <opencv2/opencv.hpp>

namespace slamplay {

inline cv::Mat depthImageToDisplayableGray(const cv::Mat &map) {
    if (map.empty()) return cv::Mat();

    cv::Mat valid_mask;
    if (map.depth() == CV_32F || map.depth() == CV_64F) {
        cv::compare(map, 0, valid_mask, cv::CMP_GT);
        cv::Mat finite_mask;
        cv::compare(map, map, finite_mask, cv::CMP_EQ);
        cv::bitwise_and(valid_mask, finite_mask, valid_mask);
    } else {
        cv::compare(map, 0, valid_mask, cv::CMP_GT);
    }

    double max = 0.0;
    cv::minMaxIdx(map, nullptr, &max, nullptr, nullptr, valid_mask);

    cv::Mat adjMap = cv::Mat::zeros(map.size(), CV_8UC1);
    if (max <= 0.0) return adjMap;

    cv::convertScaleAbs(map, adjMap, 255.0 / max);
    cv::bitwise_and(adjMap, valid_mask, adjMap);
    return adjMap;
}

void showDepthImage(const std::string &windowName, const cv::Mat &map) {
    cv::imshow(windowName, depthImageToDisplayableGray(map));
}

cv::Mat convertDepthImageToGray(const cv::Mat &map) {
    return depthImageToDisplayableGray(map);
}

void plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate, const cv::Mat &depth_variance, const double factor = 0.4, int border = 0) {
    cv::imshow("depth_truth", depth_truth * factor);
    cv::imshow("depth_estimate", depth_estimate * factor);

    const int width = depth_truth.cols;
    const int height = depth_truth.rows;
    cv::Rect roi(border, border, width - 2 * border, height - 2 * border);
    cv::Mat depth_truth_roi = cv::Mat(depth_truth, roi);
    cv::Mat depth_estimate_roi = cv::Mat(depth_estimate, roi);
    cv::Mat depth_error_roi = depth_truth_roi - depth_estimate_roi;
    cv::imshow("depth_error", depth_error_roi * factor);

    cv::imshow("depth_variance", depth_variance * factor);

#if 1
    plotImageErrorWithColorbar(depth_error_roi);
#endif

    cv::waitKey(1);
}

}  // namespace slamplay