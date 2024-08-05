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