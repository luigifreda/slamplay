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

#include "lio_slam/frontend.h"
#include "lio_slam/map_utils.h"

#include "ad/io/io_utils.h"
#include "ad/laser_3d/lio_iekf/lio_iekf.h"

#include <yaml-cpp/yaml.h>

#include "macros.h"
#include <filesystem>

namespace sad {

Frontend::Frontend(const std::string &config_yaml) {
  config_yaml_ = config_yaml;

  if (!std::filesystem::exists(resultsLioMappingPath)) {
    std::filesystem::create_directories(resultsLioMappingPath);
  }
}

bool Frontend::Init() {
  LOG(INFO) << "load yaml from " << config_yaml_;
  auto yaml = YAML::LoadFile(config_yaml_);
  try {
    auto n = yaml["bag_path"];
    LOG(INFO) << Dump(n);
    bag_path_ = yaml["bag_path"].as<std::string>();
    lio_yaml_ = yaml["lio_yaml"].as<std::string>();
  } catch (...) {
    LOG(ERROR) << "failed to parse yaml";
    return false;
  }

  system(("rm -rf " + resultsLioMappingPath + "/*.pcd").c_str());
  system(("rm -rf " + resultsLioMappingPath + "/keyframes.txt").c_str());

  LioIEKF::Options options;
  options.with_ui_ = false; // mapping does not need the frontend UI
  lio_ = std::make_shared<LioIEKF>(options);
  lio_->Init(lio_yaml_);
  return true;
}

void Frontend::Run() {
  sad::RosbagIO rosbag_io(bag_path_, DatasetType::NCLT);

  // Extract RTK poses first; NCLT provides translation only
  rosbag_io
      .AddAutoRTKHandle([this](GNSSPtr gnss) {
        gnss_.emplace(gnss->unix_time_, gnss);
        return true;
      })
      .Go();
  rosbag_io.CleanProcessFunc(); // RTK processing no longer needed

  RemoveMapOrigin();

  // Run LIO
  rosbag_io
      .AddAutoPointCloudHandle(
          [&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            lio_->PCLCallBack(cloud);
            ExtractKeyFrame(lio_->GetCurrentState());
            return true;
          })
      .AddImuHandle([&](IMUPtr imu) {
        lio_->IMUCallBack(imu);
        return true;
      })
      .Go();
  lio_->Finish();

  // Save run results
  SaveKeyframes();

  LOG(INFO) << "done.";
}

void Frontend::ExtractKeyFrame(const sad::NavStated &state) {
  if (last_kf_ == nullptr) {
    if (!lio_->GetCurrentScan()) {
      // LIO not initialized yet
      return;
    }
    // First frame
    auto kf = std::make_shared<Keyframe>(
        state.timestamp_, kf_id_++, state.GetSE3(), lio_->GetCurrentScan());
    FindGPSPose(kf);
    kf->SaveAndUnloadScan(resultsLioMappingPath + "/");
    keyframes_.emplace(kf->id_, kf);
    last_kf_ = kf;
  } else {
    // Relative motion vs last keyframe exceeds thresholds
    SE3 delta = last_kf_->lidar_pose_.inverse() * state.GetSE3();
    if (delta.translation().norm() > kf_dis_th_ ||
        delta.so3().log().norm() > kf_ang_th_deg_ * math::kDEG2RAD) {
      auto kf = std::make_shared<Keyframe>(
          state.timestamp_, kf_id_++, state.GetSE3(), lio_->GetCurrentScan());
      FindGPSPose(kf);
      keyframes_.emplace(kf->id_, kf);
      kf->SaveAndUnloadScan(resultsLioMappingPath + "/");
      LOG(INFO) << "Created keyframe " << kf->id_;
      last_kf_ = kf;
    }
  }
}

void Frontend::FindGPSPose(std::shared_ptr<Keyframe> kf) {
  SE3 pose;
  GNSSPtr match;
  if (math::PoseInterp<GNSSPtr>(
          kf->timestamp_, gnss_,
          [](const GNSSPtr &gnss) -> SE3 { return gnss->utm_pose_; }, pose,
          match)) {
    kf->rtk_pose_ = pose;
    kf->rtk_valid_ = true;
  } else {
    kf->rtk_valid_ = false;
  }
}

void Frontend::SaveKeyframes() {
  std::ofstream fout(resultsLioMappingPath + "/keyframes.txt");
  for (auto &kfp : keyframes_) {
    kfp.second->Save(fout);
  }
  fout.close();
}

void Frontend::RemoveMapOrigin() {
  if (gnss_.empty()) {
    return;
  }

  bool origin_set = false;
  for (auto &p : gnss_) {
    if (p.second->status_ == GpsStatusType::GNSS_FIXED_SOLUTION) {
      map_origin_ = p.second->utm_pose_.translation();
      origin_set = true;

      LOG(INFO) << "map origin is set to " << map_origin_.transpose();

      auto yaml = YAML::LoadFile(config_yaml_);
      std::vector<double> ori{map_origin_[0], map_origin_[1], map_origin_[2]};
      yaml["origin"] = ori;

      std::ofstream fout(config_yaml_);
      fout << yaml;
      break;
    }
  }

  if (origin_set) {
    LOG(INFO) << "removing origin from rtk";
    for (auto &p : gnss_) {
      p.second->utm_pose_.translation() -= map_origin_;
    }
  }
}

} // namespace sad