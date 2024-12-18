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

void showDepthImage(const std::string &windowName, const cv::Mat &map) {
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    cv::convertScaleAbs(map, adjMap, 255 / max);
    cv::imshow(windowName, adjMap);
}

cv::Mat convertDepthImageToGray(const cv::Mat &map) {
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    cv::convertScaleAbs(map, adjMap, 255 / max);
    return adjMap;
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