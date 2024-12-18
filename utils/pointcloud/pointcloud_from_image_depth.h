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

#include <math.h>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

#include <camera/cam_utils.h>

#include "io/messages.h"

namespace slamplay {

// get point cloud from image and distance (distance pixels represent the distance |OP|)
template <typename PointT, typename Scalar = double>
inline void getPointCloudFromImageAndDistance(const cv::Mat& color,
                                              const cv::Mat_<Scalar>& distance,
                                              const cv::Mat& maskIn,
                                              const Intrinsics& intrinsics,
                                              const int border,
                                              const Eigen::Isometry3d& T,
                                              pcl::PointCloud<PointT>& pointCloud) {
    pointCloud.clear();
    cv::Mat mask = maskIn.empty() ? cv::Mat(color.size(), CV_8U, 1) : maskIn;
    MSG_ASSERT(mask.type() == CV_8U || mask.type() == CV_8UC1, "Must be a mask!");

    const auto& fx = intrinsics.fx;
    const auto& fy = intrinsics.fy;
    const auto& cx = intrinsics.cx;
    const auto& cy = intrinsics.cy;

    for (int v = border; v < color.rows - border; v++)
    {
        const Scalar* distance_ptr_v = distance.template ptr<Scalar>(v);
        const uchar* mask_ptr_v = mask.ptr<uchar>(v);
        for (int u = border; u < color.cols - border; u++)
        {
            const Scalar distance = distance_ptr_v[u];  // distance along the ray
            const uchar valid = mask_ptr_v[u];
            if (distance == 0 || valid == 0)
                continue;  // distance 0 means not measured
            Eigen::Vector3d point;
            point[0] = (u - cx) / fx;
            point[1] = (v - cy) / fy;
            point[2] = 1;
            point.normalize();
            point *= distance;
            Eigen::Vector3d pointWorld = T * point;

            PointT p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            p.b = color.data[v * color.step + u * color.channels()];
            p.g = color.data[v * color.step + u * color.channels() + 1];
            p.r = color.data[v * color.step + u * color.channels() + 2];  // red
            pointCloud.points.push_back(p);
        }
    }

    pointCloud.width = pointCloud.size();
    pointCloud.height = 1;
}

// get point cloud from image and depth
template <typename PointT, typename Scalar = double>
inline void getPointCloudFromImageAndDepth(const cv::Mat& color,
                                           const cv::Mat_<Scalar>& depth,
                                           const cv::Mat& maskIn,
                                           const Intrinsics& intrinsics,
                                           const int border,
                                           const Eigen::Isometry3d& T,
                                           pcl::PointCloud<PointT>& pointCloud) {
    pointCloud.clear();
    cv::Mat mask = maskIn.empty() ? cv::Mat(color.size(), CV_8U, 1) : maskIn;
    MSG_ASSERT(mask.type() == CV_8U || mask.type() == CV_8UC1, "Must be a mask!");

    const auto& fx = intrinsics.fx;
    const auto& fy = intrinsics.fy;
    const auto& cx = intrinsics.cx;
    const auto& cy = intrinsics.cy;

    for (int v = border; v < color.rows - border; v++)
    {
        const Scalar* depth_ptr_v = depth.template ptr<Scalar>(v);
        const uchar* mask_ptr_v = mask.ptr<uchar>(v);
        for (int u = border; u < color.cols - border; u++)
        {
            const Scalar depth = depth_ptr_v[u];  // z depth
            const uchar valid = mask_ptr_v[u];
            if (depth == 0 || valid == 0)
                continue;  // 0 means not measured
            Eigen::Vector3d point;
            point[0] = depth * (u - cx) / fx;
            point[1] = depth * (v - cy) / fy;
            point[2] = depth;
            Eigen::Vector3d pointWorld = T * point;

            PointT p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            p.b = color.data[v * color.step + u * color.channels()];
            p.g = color.data[v * color.step + u * color.channels() + 1];
            p.r = color.data[v * color.step + u * color.channels() + 2];  // red
            pointCloud.points.push_back(p);
        }
    }

    pointCloud.width = pointCloud.size();
    pointCloud.height = 1;
}

}  // namespace slamplay