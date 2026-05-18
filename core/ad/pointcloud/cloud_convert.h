#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "ad/pointcloud/point_types.h"

namespace sad {

/**
 * Preprocess LiDAR point clouds
 *
 * Convert Velodyne, Ouster, Avia data to FullCloud format
 * This class is held by MessageSync, responsible for synchronizing received LiDAR messages
 * with IMU and preprocessing them before passing to the LO/LIO algorithm
 */
class CloudConvert {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class LidarType {
    AVIA = 1, // DJI solid-state LiDAR
    VELO32,   // Velodyne 32-beam
    OUST64,   // Ouster 64-beam
  };

  CloudConvert() = default;
  ~CloudConvert() = default;

  /**
   * Process Livox Avia point cloud
   * @param msg
   * @param pcl_out
   */
  void Process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
               FullCloudPtr &pcl_out);

  /**
   * Process sensor_msgs::PointCloud2 point cloud
   * @param msg
   * @param pcl_out
   */
  void Process(const sensor_msgs::PointCloud2::ConstPtr &msg,
               FullCloudPtr &pcl_out);

  /// Load parameters from YAML
  void LoadFromYAML(const std::string &yaml);

private:
  void AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg);

  FullPointCloudType cloud_full_, cloud_out_; // Output point cloud
  LidarType lidar_type_ = LidarType::AVIA;    // LiDAR type
  int point_filter_num_ = 1;                  // Point skip factor
  int num_scans_ = 6;                         // Number of scan lines
  float time_scale_ = 1e-3;                   // Ratio of point time field to seconds
};
} // namespace sad