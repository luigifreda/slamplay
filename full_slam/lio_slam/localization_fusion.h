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
//
// Created by gaoxiang
// From https://github.com/gaoxiang12/slam_in_autonomous_driving
// Modified by Luigi Freda later for slamplay
//

#ifndef SLAM_IN_AUTO_DRIVING_FUSION_H
#define SLAM_IN_AUTO_DRIVING_FUSION_H

#include "ad/common/eigen_types.h"
#include "ad/common/message_def.h"
#include "ad/imu/imu.h"
#include "ad/imu/static_imu_init.h"
#include "ad/kf/eskf.hpp"
#include "ad/nav/gnss.h"
#include "ad/pointcloud/point_types.h"

#include "ad/laser_3d/message_sync/message_sync.h"
#include "ad/pointcloud/cloud_convert.h"

#include "viz/ad/autonomous_driving_viz.h"

#include <pcl/registration/ndt.h>
#include <sensor_msgs/PointCloud2.h>

namespace sad {

/**
 * High-precision fusion localization: fuses IMU, RTK, and lidar.
 *
 * NOTE: Some IMU outlier handling is not implemented here; bad IMU can bias the
 * estimate.
 */
class LocalizationFusion {
public:
  explicit LocalizationFusion(const std::string &config_yaml);

  enum class Status {
    WAITING_FOR_RTK, // waiting for initial RTK fix
    WORKING,         // normal operation
  };

  /// Initialize and load parameters
  bool Init();

  /// Process incoming RTK
  void ProcessRTK(GNSSPtr gnss);
  void ProcessIMU(IMUPtr imu);
  void ProcessPointCloud(sensor_msgs::PointCloud2::Ptr cloud);

private:
  /// Load map tiles near a pose
  void LoadMap(const SE3 &pose);

  /// Process synchronized IMU and lidar data
  void ProcessMeasurements(const MeasureGroup &meas);

  /// Load map tile index file
  void LoadMapIndex();

  /// Grid search result
  struct GridSearchResult {
    SE3 pose_;
    SE3 result_pose_;
    double score_ = 0.0;
  };

  /// Search vehicle pose near initial RTK position
  bool SearchRTK();

  /// Align one grid-search pose; returns pose and score
  void AlignForGrid(GridSearchResult &gr);

  /// Lidar localization
  bool LidarLocalization();

  /// Initialize using IMU
  void TryInitIMU();

  /// Predict state with IMU; predictions stored in imu_states_
  void Predict();

  /// Undistort point cloud in measures_
  void Undistort();

  /// Run alignment and observation update
  void Align();

  /// State
  Status status_ = Status::WAITING_FOR_RTK;

  /// Data
  std::string config_yaml_;                         // config file path
  Vec3d map_origin_ = Vec3d::Zero();                // map origin
  std::string data_path_;                           // map data directory
  std::set<Vec2i, less_vec<2>> map_data_index_;     // grid cells with map data
  std::map<Vec2i, CloudPtr, less_vec<2>> map_data_; // map built in Ch. 9

  std::shared_ptr<MessageSync> sync_ = nullptr; // message synchronizer
  StaticIMUInit imu_init_;                      // static IMU initialization

  /// Filter
  ESKFD eskf_;
  std::vector<NavStated> imu_states_; // states during ESKF prediction

  FullCloudPtr scan_undistort_{
      new FullPointCloudType()}; // scan after undistortion
  CloudPtr current_scan_ = nullptr;

  SE3 TIL_;
  MeasureGroup measures_; // synchronized IMU and lidar scan
  GNSSPtr last_gnss_ = nullptr;

  bool init_has_failed_ = false; // whether initialization has failed before
  SE3 last_searched_pos_;        // last GNSS position used for search

  /// Lidar localization
  bool imu_need_init_ = true;    // whether IMU bias init is still needed
  CloudPtr ref_cloud_ = nullptr; // reference cloud for NDT
  pcl::NormalDistributionsTransform<PointType, PointType> ndt_;

  /// Parameters
  double rtk_search_min_score_ = 4.5;

  // ui
  std::shared_ptr<ui::AutonomousDrivingViz> ui_ = nullptr;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_FUSION_H
