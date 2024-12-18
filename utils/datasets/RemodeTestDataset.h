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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace slamplay {

bool readRemodeDatasetFiles(
    const std::string &path,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth,
    int width = 640,
    int height = 480) {
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: image file name tx, ty, tz, qx, qy, qz, qw, note that it is TWC instead of TCW
        std::string image;
        fin >> image;
        double data[7];
        for (double &d : data) fin >> d;

        color_image_files.push_back(path + std::string("/images/") + image);
        poses.push_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                         Eigen::Vector3d(data[0], data[1], data[2])));
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

}  // namespace slamplay