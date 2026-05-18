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
#ifndef SLAM_IN_AUTO_DRIVING_OPTIMIZATION_H
#define SLAM_IN_AUTO_DRIVING_OPTIMIZATION_H

#include <g2o/core/sparse_optimizer.h>

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "ad/g2o/g2o_types.h"
#include "lio_slam/keyframe.h"
#include "lio_slam/loop_closure.h"

namespace sad {

// Backend optimization
class Optimization {
public:
  explicit Optimization(const std::string &yaml);

  /// Initialize; stage 1 or stage 2
  bool Init(int stage = 1);

  /// Run optimization
  void Run();

private:
  /// If RTK has no rotation throughout, align lidar and RTK trajectories first
  void InitialAlign();

  /// Build the optimization problem
  void BuildProblem();

  /// Add vertices
  void AddVertices();

  /// Add pose observations
  void AddRTKEdges();
  void AddLidarEdges();
  void AddLoopEdges();

  /// Solve the problem
  void Solve();

  /// Remove outliers
  void RemoveOutliers();

  /// Save optimization results
  void SaveResults();

  /// Load loop closure candidates
  void LoadLoopCandidates();

  /// Save g2o file
  void SaveG2O(const std::string &file_name);

  std::string yaml_;
  std::map<IdType, KFPtr> keyframes_;
  bool rtk_has_rot_ = false;
  int stage_ = 1; // optimization stage
  SE3 TBG_;       // body-to-GNSS extrinsic

  std::vector<LoopCandidate> loop_candidates_; // loop closure candidates

  std::map<IdType, VertexPose *> vertices_;

  g2o::SparseOptimizer optimizer_;
  std::vector<EdgeGNSS *> gnss_edge_;
  std::vector<EdgeGNSSTransOnly *> gnss_trans_edge_;

  std::vector<EdgeRelativeMotion *> lidar_edge_;
  std::vector<EdgeRelativeMotion *> loop_edge_;

  // Parameters
  double rtk_outlier_th_ = 1.0;  // RTK outlier threshold
  int lidar_continuous_num_ = 3; // number of consecutive lidar pose edges
  double rtk_pos_noise_ = 0.5;
  double rtk_ang_noise_ = 2.0 * math::kDEG2RAD;
  double rtk_height_noise_ratio_ = 20.0;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_OPTIMIZATION_H
