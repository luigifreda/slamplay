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
#include <pangolin/pangolin.h>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "DirectPoseEstimation.h"
#include "image/image_pyramids.h"
#include "image/image_utils.h"
#include "macros.h"

using namespace std;
using namespace cv;
using namespace slamplay;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// input images
string left_file = dataDir + "/opticflow/left.png";
string disparity_file = dataDir + "/opticflow/disparity.png";
boost::format fmt_others(dataDir + "/opticflow/%06d.png");  // other files

int main(int argc, char **argv) {
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    Intrinsics intrinsics{fx, fy, cx, cy};
    DirectMethodParams params;

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int border = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(border, left_img.cols - border);  // don't pick pixels close to border
        int y = rng.uniform(border, left_img.rows - border);  // don't pick pixels close to border
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity;  // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    const int num_images = 6;
    for (int i = 1; i < num_images; i++)
    {
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        cv::Mat img2_show;
#if 0       
        // try single layer by uncommenting this 
        DirectPoseEstimationSingleLayer(intrinsics, params, left_img, img, pixels_ref, depth_ref, T_cur_ref, img2_show);
#else
        std::vector<cv::Mat> pyr2_show;
        DirectPoseEstimationMultiLayer(intrinsics, params, left_img, img, pixels_ref, depth_ref, T_cur_ref, pyr2_show);
        composePyrImage(pyr2_show, img2_show);
#endif
        cv::imshow("current", img2_show);
        cv::waitKey();
    }
    return 0;
}