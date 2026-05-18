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
#ifndef SLAM_IN_AUTO_DRIVING_FRONTEND_H
#define SLAM_IN_AUTO_DRIVING_FRONTEND_H

#include <map>
#include <memory>
#include <string>

#include "ad/nav/gnss.h"
#include "ad/nav/nav_state.h"

#include "lio_slam/keyframe.h"

namespace sad {

class LioIEKF;
/**
 * Mapping frontend: feeds IMU and lidar to LIO, parses RTK into rtk_pose
 */
class Frontend {
public:
  struct Options {};

  // Load data paths from yaml
  explicit Frontend(const std::string &config_yaml);

  // Init: create LIO and check data availability
  bool Init();

  /// Run the frontend
  void Run();

private:
  /// Extract a keyframe from state when motion exceeds thresholds
  void ExtractKeyFrame(const NavStated &state);

  /// Assign GPS/RTK pose to a keyframe
  void FindGPSPose(std::shared_ptr<Keyframe> kf);

  /// Save keyframe poses (scans are saved when created)
  void SaveKeyframes();

  /// Set map origin from RTK and subtract it from all GNSS poses
  void RemoveMapOrigin();

  // Data
  std::shared_ptr<Keyframe> last_kf_ = nullptr; // most recent keyframe
  std::map<IdType, std::shared_ptr<Keyframe>> keyframes_; // extracted keyframes
  std::shared_ptr<LioIEKF> lio_ = nullptr;                // LIO
  std::string config_yaml_;                               // config file path

  std::map<double, GNSSPtr> gnss_; // GNSS data
  IdType kf_id_ = 0;               // latest keyframe ID
  Vec3d map_origin_ = Vec3d::Zero();

  // Parameters and configuration
  std::string bag_path_;        // rosbag path
  std::string lio_yaml_;        // LIO config yaml
  double kf_dis_th_ = 1.0;      // keyframe distance threshold
  double kf_ang_th_deg_ = 10.0; // keyframe angle threshold (degrees)
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_FRONTEND_H
