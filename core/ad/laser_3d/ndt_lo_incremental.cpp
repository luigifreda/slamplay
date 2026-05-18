#include "ad/laser_3d/ndt_lo_incremental.h"
#include "ad/common/math_utils.h"
#include "ad/timer/timer.h"

namespace sad {

void NDTLOIncremental::AddCloud(CloudPtr scan, SE3 &pose, bool use_guess) {
  if (first_frame_) {
    // first frame, add directly to the local map
    pose = SE3();
    last_kf_pose_ = pose;
    ndt_.AddCloud(scan);
    first_frame_ = false;
    return;
  }

  // at this point the local map is inside NDT, align directly
  SE3 guess;
  ndt_.SetSource(scan);
  if (estimated_poses_.size() < 2) {
    ndt_.AlignNdt(guess);
  } else {
    if (!use_guess) {
      // extrapolate from the two most recent poses
      SE3 T1 = estimated_poses_[estimated_poses_.size() - 1];
      SE3 T2 = estimated_poses_[estimated_poses_.size() - 2];
      guess = T1 * (T2.inverse() * T1);
    } else {
      guess = pose;
    }

    ndt_.AlignNdt(guess);
  }

  pose = guess;
  estimated_poses_.emplace_back(pose);

  CloudPtr scan_world(new PointCloudType);
  pcl::transformPointCloud(*scan, *scan_world, guess.matrix().cast<float>());

  if (IsKeyframe(pose)) {
    last_kf_pose_ = pose;
    cnt_frame_ = 0;
    // add to the local map inside NDT
    ndt_.AddCloud(scan_world);
  }

  if (viewer_ != nullptr) {
    viewer_->SetPoseAndCloud(pose, scan_world);
  }
  cnt_frame_++;
}

bool NDTLOIncremental::IsKeyframe(const SE3 &current_pose) {
  if (cnt_frame_ > 10) {
    return true;
  }

  SE3 delta = last_kf_pose_.inverse() * current_pose;
  return delta.translation().norm() > options_.kf_distance_ ||
         delta.so3().log().norm() > options_.kf_angle_deg_ * math::kDEG2RAD;
}

void NDTLOIncremental::SaveMap(const std::string &map_path) {
  if (viewer_) {
    viewer_->SaveMap(map_path);
  }
}

} // namespace sad