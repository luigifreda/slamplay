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

    std::cout << "starting here" << std::endl;

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