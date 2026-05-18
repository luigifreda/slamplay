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
#ifndef SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H
#define SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include "lio_slam/keyframe.h"

namespace sad {

/**
 * Loop closure candidate
 */
struct LoopCandidate {
  LoopCandidate() {}
  LoopCandidate(IdType id1, IdType id2, SE3 Tij)
      : idx1_(id1), idx2_(id2), Tij_(Tij) {}

  IdType idx1_ = 0;
  IdType idx2_ = 0;
  SE3 Tij_;
  double ndt_score_ = 0.0;
};

class LoopClosure {
public:
  explicit LoopClosure(const std ::string &config_yaml);

  bool Init();

  void Run();

private:
  /// Detect loop closure candidates
  void DetectLoopCandidates();

  /// Compute relative motion for loop candidates
  void ComputeLoopCandidates();

  /// Verify a single loop candidate
  void ComputeForCandidate(LoopCandidate &c);

  /// Save results
  void SaveResults();

  /// params
  std::vector<LoopCandidate> loop_candiates_;
  int min_id_interval_ = 50;  // min ID gap between candidate keyframes
  double min_distance_ = 30;  // min distance between candidates
  int skip_id_ = 5;           // IDs to skip after selecting a candidate
  double ndt_score_th_ = 2.5; // NDT score threshold for a valid loop

  std::map<IdType, KFPtr> keyframes_;

  std::string yaml_;
};
} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H
