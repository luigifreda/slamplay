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
#ifndef SLAM_IN_AUTO_DRIVING_KEYFRAME_H
#define SLAM_IN_AUTO_DRIVING_KEYFRAME_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include <map>

namespace sad {

struct Keyframe {
  Keyframe() {}
  Keyframe(double time, IdType id, const SE3 &lidar_pose, CloudPtr cloud)
      : timestamp_(time), id_(id), lidar_pose_(lidar_pose), cloud_(cloud) {}

  /// Save scan to disk and release from memory
  void SaveAndUnloadScan(const std::string &path);

  void LoadScan(const std::string &path);

  /// Save to text file
  void Save(std::ostream &os);

  /// Load from file
  void Load(std::istream &is);

  double timestamp_ = 0;           // timestamp
  IdType id_ = 0;                  // unique keyframe id
  SE3 lidar_pose_;                 // lidar pose
  SE3 rtk_pose_;                   // RTK pose
  SE3 opti_pose_1_;                // stage-1 optimized pose
  SE3 opti_pose_2_;                // stage-2 optimized pose
  bool rtk_heading_valid_ = false; // whether RTK includes heading
  bool rtk_valid_ = true;          // whether raw RTK is valid
  bool rtk_inlier_ = true;         // whether RTK is an inlier in optimization

  CloudPtr cloud_ = nullptr;
};

bool LoadKeyFrames(const std::string &path,
                   std::map<IdType, std::shared_ptr<Keyframe>> &keyframes);
} // namespace sad

using KFPtr = std::shared_ptr<sad::Keyframe>;

#endif // SLAM_IN_AUTO_DRIVING_KEYFRAME_H
