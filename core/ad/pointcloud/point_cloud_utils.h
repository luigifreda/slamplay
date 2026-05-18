//
// Created by xiang on 2021/8/25.
//

#ifndef SLAM_IN_AUTO_DRIVING_POINT_CLOUD_UTILS_H
#define SLAM_IN_AUTO_DRIVING_POINT_CLOUD_UTILS_H

#include "ad/pointcloud/point_types.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// Utility functions for point clouds

namespace sad {

/// Voxel filtering
void VoxelGrid(CloudPtr cloud, float voxel_size = 0.05);

/// Remove ground points
void RemoveGround(CloudPtr cloud, float z_min = 0.5);

/// Write point cloud file
template <typename CloudType>
void SaveCloudToFile(const std::string &filePath, CloudType &cloud);

/// Convert a point cloud into a bird's-eye view image
template <typename PointCloudType>
cv::Mat GenerateBEVImage(typename PointCloudType::Ptr cloud,
                         double image_resolution, double min_z, double max_z) {

  using PointType = typename PointCloudType::PointType;

  // Compute point cloud bounds
  auto minmax_x = std::minmax_element(
      cloud->points.begin(), cloud->points.end(),
      [](const PointType &p1, const PointType &p2) { return p1.x < p2.x; });
  auto minmax_y = std::minmax_element(
      cloud->points.begin(), cloud->points.end(),
      [](const PointType &p1, const PointType &p2) { return p1.y < p2.y; });
  double min_x = minmax_x.first->x;
  double max_x = minmax_x.second->x;
  double min_y = minmax_y.first->y;
  double max_y = minmax_y.second->y;

  const double inv_r = 1.0 / image_resolution;

  const int image_rows = int((max_y - min_y) * inv_r);
  const int image_cols = int((max_x - min_x) * inv_r);

  float x_center = 0.5 * (max_x + min_x);
  float y_center = 0.5 * (max_y + min_y);
  float x_center_image = image_cols / 2;
  float y_center_image = image_rows / 2;

  // Generate image
  cv::Mat image(image_rows, image_cols, CV_8UC3, cv::Scalar(255, 255, 255));

  for (const auto &pt : cloud->points) {
    int x = int((pt.x - x_center) * inv_r + x_center_image);
    int y = int((pt.y - y_center) * inv_r + y_center_image);
    if (x < 0 || x >= image_cols || y < 0 || y >= image_rows || pt.z < min_z ||
        pt.z > max_z) {
      continue;
    }

    image.at<cv::Vec3b>(y, x) = cv::Vec3b(227, 143, 79);
  }

  return image;
}

template <typename PointCloudType>
cv::Mat GenerateRangeImage(typename PointCloudType::Ptr cloud,
                           double azimuth_resolution, int elevation_rows,
                           double elevation_range, double lidar_height) {
  // using PointType = typename PointCloudType::PointType;

  const int image_cols = int(
      360 / azimuth_resolution); // 360 degrees horizontal, split by resolution
  const int image_rows = elevation_rows; // fixed number of image rows

  // Generate an HSV image for better visualization
  cv::Mat image(image_rows, image_cols, CV_8UC3, cv::Scalar(0, 0, 0));
  const double ele_resolution =
      elevation_range * 2 / elevation_rows; // elevation resolution

  for (const auto &pt : cloud->points) {
    double azimuth = atan2(pt.y, pt.x) * 180 / M_PI;
    double range = sqrt(pt.x * pt.x + pt.y * pt.y);
    double elevation = asin((pt.z - lidar_height) / range) * 180 / M_PI;

    // keep in 0~360
    if (azimuth < 0) {
      azimuth += 360;
    }

    int x = int(azimuth / azimuth_resolution);                         // column
    int y = int((elevation + elevation_range) / ele_resolution + 0.5); // row

    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
      image.at<cv::Vec3b>(y, x) =
          cv::Vec3b(uchar(range / 100 * 255.0), 255, 127);
    }
  }

  // Flip along Y axis so that Y points up when Z points up
  cv::Mat image_flipped;
  cv::flip(image, image_flipped, 0);

  // hsv to rgb
  cv::Mat image_rgb;
  cv::cvtColor(image_flipped, image_rgb, cv::COLOR_HSV2RGB);
  return image_rgb;
}

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_POINT_CLOUD_UTILS_H
