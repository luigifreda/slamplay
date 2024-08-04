#pragma once

#include <opencv2/opencv.hpp>

std::vector<cv::KeyPoint> undistortVectorOfKeyPoints(const std::vector<cv::KeyPoint> &src, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    std::vector<cv::Point2f> mat;
    std::vector<cv::KeyPoint> res;
    for (auto temp : src)
    {
        mat.emplace_back(temp.pt);
    }
    cv::undistortPoints(mat, mat, cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix);
    for (int index = 0; index < mat.size(); ++index)
    {
        auto kpt = src[index];
        kpt.pt = mat[index];
        res.emplace_back(kpt);
    }
    return res;
}
