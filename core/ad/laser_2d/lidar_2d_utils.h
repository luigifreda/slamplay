#ifndef SLAM_IN_AUTO_DRIVING_LIDAR_2D_UTILS_H
#define SLAM_IN_AUTO_DRIVING_LIDAR_2D_UTILS_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/lidar_utils.h"

#include <opencv2/core/core.hpp>

/// Utility functions for 2D lidar
namespace sad {

/**
 * Draw a 2D scan on image
 * @param scan
 * @param pose
 * @param image
 * @param image_size image size
 * @param resolution resolution, pixels per meter
 * @param pose_submap if using a submap, provide the submap's pose
 */
void Visualize2DScan(Scan2d::Ptr scan, const SE2 &pose, cv::Mat &image,
                     const Vec3b &color, int image_size = 800,
                     float resolution = 20.0, const SE2 &pose_submap = SE2());

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_LIDAR_2D_UTILS_H
