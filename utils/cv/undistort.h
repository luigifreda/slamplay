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

#include <opencv2/opencv.hpp>

namespace slamplay {

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

}