#include "ad/laser_3d/ndt_lo_direct.h"
#include "ad/common/math_utils.h"
#include "viz/ad/pcl_map_viewer.h"

#include <pcl/common/transforms.h>

namespace sad {

void NDTLODirect::AddCloud(CloudPtr scan, SE3 &pose) {
  if (local_map_ == nullptr) {
    // first frame, add directly to the local map
    local_map_.reset(new PointCloudType);
    // operator += concatenates point clouds
    *local_map_ += *scan;
    pose = SE3();
    last_kf_pose_ = pose;

    if (options_.use_pcl_ndt_) {
      ndt_pcl_.setInputTarget(local_map_);
    } else {
      ndt_.SetTarget(local_map_);
    }

    return;
  }

  // compute pose of scan relative to local map
  pose = AlignWithLocalMap(scan);
  CloudPtr scan_world(new PointCloudType);
  pcl::transformPointCloud(*scan, *scan_world, pose.matrix().cast<float>());

  if (IsKeyframe(pose)) {
    last_kf_pose_ = pose;

    // rebuild local map
    scans_in_local_map_.emplace_back(scan_world);
    if (scans_in_local_map_.size() > options_.num_kfs_in_local_map_) {
      scans_in_local_map_.pop_front();
    }

    local_map_.reset(new PointCloudType);
    for (auto &scan : scans_in_local_map_) {
      *local_map_ += *scan;
    }

    if (options_.use_pcl_ndt_) {
      ndt_pcl_.setInputTarget(local_map_);
    } else {
      ndt_.SetTarget(local_map_);
    }
  }

  if (viewer_ != nullptr) {
    viewer_->SetPoseAndCloud(pose, scan_world);
  }
}

bool NDTLODirect::IsKeyframe(const SE3 &current_pose) {
  // mark as keyframe if relative motion from last frame exceeds a distance or
  // angle threshold
  SE3 delta = last_kf_pose_.inverse() * current_pose;
  return delta.translation().norm() > options_.kf_distance_ ||
         delta.so3().log().norm() > options_.kf_angle_deg_ * math::kDEG2RAD;
}

SE3 NDTLODirect::AlignWithLocalMap(CloudPtr scan) {
  if (options_.use_pcl_ndt_) {
    ndt_pcl_.setInputSource(scan);
  } else {
    ndt_.SetSource(scan);
  }

  CloudPtr output(new PointCloudType());

  SE3 guess;
  bool align_success = true;
  if (estimated_poses_.size() < 2) {
    if (options_.use_pcl_ndt_) {
      ndt_pcl_.align(*output, guess.matrix().cast<float>());
      guess =
          Mat4ToSE3(ndt_pcl_.getFinalTransformation().cast<double>().eval());
    } else {
      align_success = ndt_.AlignNdt(guess);
    }
  } else {
    // extrapolate from the two most recent poses
    SE3 T1 = estimated_poses_[estimated_poses_.size() - 1];
    SE3 T2 = estimated_poses_[estimated_poses_.size() - 2];
    guess = T1 * (T2.inverse() * T1);

    if (options_.use_pcl_ndt_) {
      ndt_pcl_.align(*output, guess.matrix().cast<float>());
      guess =
          Mat4ToSE3(ndt_pcl_.getFinalTransformation().cast<double>().eval());
    } else {
      align_success = ndt_.AlignNdt(guess);
    }
  }

  LOG(INFO) << "pose: " << guess.translation().transpose() << ", "
            << guess.so3().unit_quaternion().coeffs().transpose();

  if (options_.use_pcl_ndt_) {
    LOG(INFO) << "trans prob: " << ndt_pcl_.getTransformationProbability();
  }

  estimated_poses_.emplace_back(guess);
  return guess;
}

void NDTLODirect::SaveMap(const std::string &map_path) {
  if (viewer_) {
    viewer_->SaveMap(map_path);
  }
}

} // namespace sad