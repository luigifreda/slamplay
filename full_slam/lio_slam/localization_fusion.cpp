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
#include <execution>
#include <yaml-cpp/yaml.h>

#include "ad/pointcloud/lidar_utils.h"
#include "lio_slam/localization_fusion.h"

#include <pcl/io/pcd_io.h>

namespace sad {

LocalizationFusion::LocalizationFusion(const std::string &config_yaml) {
  config_yaml_ = config_yaml;
  StaticIMUInit::Options imu_init_options;
  imu_init_options.use_speed_for_static_checking_ =
      false; // wheel odometry not needed for this dataset
  imu_init_ = StaticIMUInit(imu_init_options);
  ndt_.setResolution(1.0);
}

bool LocalizationFusion::Init() {
  // Map origin
  auto yaml = YAML::LoadFile(config_yaml_);
  auto origin_data = yaml["origin"].as<std::vector<double>>();
  map_origin_ = Vec3d(origin_data[0], origin_data[1], origin_data[2]);

  // Map data directory
  data_path_ = yaml["map_data"].as<std::string>();
  LoadMapIndex();

  // Lidar and IMU message sync
  sync_ = std::make_shared<MessageSync>(
      [this](const MeasureGroup &m) { ProcessMeasurements(m); });
  sync_->Init(config_yaml_);

  // Lidar-IMU extrinsics
  std::vector<double> ext_t =
      yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
  std::vector<double> ext_r =
      yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();
  Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
  Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
  TIL_ = SE3(lidar_R_wrt_IMU, lidar_T_wrt_IMU);

  // ui
  ui_ = std::make_shared<ui::AutonomousDrivingViz>();
  ui_->Init();
  ui_->SetCurrentScanSize(50);
  return true;
}

void LocalizationFusion::ProcessRTK(GNSSPtr gnss) {
  gnss->utm_pose_.translation() -= map_origin_; // subtract map origin
  last_gnss_ = gnss;
}

void LocalizationFusion::ProcessMeasurements(const MeasureGroup &meas) {
  measures_ = meas;

  if (imu_need_init_) {
    TryInitIMU();
    return;
  }

  /// Same three steps as LIO; Align performs map matching
  if (status_ == Status::WORKING) {
    Predict();
    Undistort();
  } else {
    scan_undistort_ = measures_.lidar_;
  }

  Align();
}

void LocalizationFusion::TryInitIMU() {
  for (auto imu : measures_.imu_) {
    imu_init_.AddIMU(*imu);
  }

  if (imu_init_.InitSuccess()) {
    // Read initial biases and configure ESKF
    sad::ESKFD::Options options;
    // Noise estimated by the initializer
    // options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
    // options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
    options.update_bias_acce_ = false;
    options.update_bias_gyro_ = false;
    eskf_.SetInitialConditions(options, imu_init_.GetInitBg(),
                               imu_init_.GetInitBa(), imu_init_.GetGravity());
    imu_need_init_ = false;

    LOG(INFO) << "IMU initialization succeeded";
  }
}

void LocalizationFusion::Predict() {
  imu_states_.clear();
  imu_states_.emplace_back(eskf_.GetNominalState());

  /// Predict IMU states
  for (auto &imu : measures_.imu_) {
    eskf_.Predict(*imu);
    imu_states_.emplace_back(eskf_.GetNominalState());
  }
}

void LocalizationFusion::Undistort() {
  auto cloud = measures_.lidar_;
  auto imu_state = eskf_.GetNominalState(); // state at scan end time
  SE3 T_end = SE3(imu_state.R_, imu_state.p_);

  /// Transform all points to the scan end-time pose
  std::for_each(std::execution::par_unseq, cloud->points.begin(),
                cloud->points.end(), [&](auto &pt) {
                  SE3 Ti = T_end;
                  NavStated match;

                  // pt.time is ms offset from scan start; interpolate pose at
                  // hit time
                  math::PoseInterp<NavStated>(
                      measures_.lidar_begin_time_ + pt.time * 1e-3, imu_states_,
                      [](const NavStated &s) { return s.timestamp_; },
                      [](const NavStated &s) { return s.GetSE3(); }, Ti, match);

                  Vec3d pi = ToVec3d(pt);
                  Vec3d p_compensate =
                      TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pi;

                  pt.x = p_compensate(0);
                  pt.y = p_compensate(1);
                  pt.z = p_compensate(2);
                });
  scan_undistort_ = cloud;
}

void LocalizationFusion::Align() {
  FullCloudPtr scan_undistort_trans(new FullPointCloudType);
  pcl::transformPointCloud(*scan_undistort_, *scan_undistort_trans,
                           TIL_.matrix());
  scan_undistort_ = scan_undistort_trans;
  current_scan_ = ConvertToCloud<FullPointType>(scan_undistort_);
  current_scan_ = VoxelCloud(current_scan_, 0.5);

  if (status_ == Status::WAITING_FOR_RTK) {
    // If recent RTK is available, try initialization
    if (last_gnss_ != nullptr) {
      if (SearchRTK()) {
        status_ == Status::WORKING;
        ui_->UpdateScan(current_scan_, eskf_.GetNominalSE3());
        ui_->UpdateNavState(eskf_.GetNominalState());
      }
    }
  } else {
    LidarLocalization();
    ui_->UpdateScan(current_scan_, eskf_.GetNominalSE3());
    ui_->UpdateNavState(eskf_.GetNominalState());
  }
}

bool LocalizationFusion::SearchRTK() {
  if (init_has_failed_) {
    if ((last_gnss_->utm_pose_.translation() - last_searched_pos_.translation())
            .norm() < 20.0) {
      LOG(INFO) << "skip this position";
      return false;
    }
  }

  // RTK has no heading; search over a range of yaw angles first
  std::vector<GridSearchResult> search_poses;
  LoadMap(last_gnss_->utm_pose_);

  /// RTK has no heading; scan yaw at fixed step
  double grid_ang_range = 360.0,
         grid_ang_step = 10; // yaw search range and step
  for (double ang = 0; ang < grid_ang_range; ang += grid_ang_step) {
    SE3 pose(SO3::rotZ(ang * math::kDEG2RAD),
             Vec3d(0, 0, 0) + last_gnss_->utm_pose_.translation());
    GridSearchResult gr;
    gr.pose_ = pose;
    search_poses.emplace_back(gr);
  }

  LOG(INFO) << "grid search poses: " << search_poses.size();
  std::for_each(std::execution::par_unseq, search_poses.begin(),
                search_poses.end(),
                [this](GridSearchResult &gr) { AlignForGrid(gr); });

  // Pick the best match
  auto max_ele = std::max_element(
      search_poses.begin(), search_poses.end(),
      [](const auto &g1, const auto &g2) { return g1.score_ < g2.score_; });
  LOG(INFO) << "max score: " << max_ele->score_ << ", pose: \n"
            << max_ele->result_pose_.matrix();
  if (max_ele->score_ > rtk_search_min_score_) {
    LOG(INFO) << "Initialization succeeded, score: " << max_ele->score_ << ">"
              << rtk_search_min_score_;
    status_ = Status::WORKING;

    /// Reset filter state
    auto state = eskf_.GetNominalState();
    state.R_ = max_ele->result_pose_.so3();
    state.p_ = max_ele->result_pose_.translation();
    state.v_.setZero();
    eskf_.SetX(state, eskf_.GetGravity());

    ESKFD::Mat18T cov;
    cov = ESKFD::Mat18T::Identity() * 1e-4;
    cov.block<12, 12>(6, 6) = Eigen::Matrix<double, 12, 12>::Identity() * 1e-6;
    eskf_.SetCov(cov);

    return true;
  }

  init_has_failed_ = true;
  last_searched_pos_ = last_gnss_->utm_pose_;
  return false;
}

void LocalizationFusion::AlignForGrid(
    sad::LocalizationFusion::GridSearchResult &gr) {
  /// Multi-resolution alignment
  pcl::NormalDistributionsTransform<PointType, PointType> ndt;
  ndt.setTransformationEpsilon(0.05);
  ndt.setStepSize(0.7);
  ndt.setMaximumIterations(40);

  ndt.setInputSource(current_scan_);
  auto map = ref_cloud_;

  CloudPtr output(new PointCloudType);
  std::vector<double> res{10.0, 5.0, 4.0, 3.0};
  Mat4f T = gr.pose_.matrix().cast<float>();
  for (auto &r : res) {
    auto rough_map = VoxelCloud(map, r * 0.1);
    ndt.setInputTarget(rough_map);
    ndt.setResolution(r);
    ndt.align(*output, T);
    T = ndt.getFinalTransformation();
  }

  gr.score_ = ndt.getTransformationProbability();
  gr.result_pose_ = Mat4ToSE3(ndt.getFinalTransformation());
}

bool LocalizationFusion::LidarLocalization() {
  SE3 pred = eskf_.GetNominalSE3();
  LoadMap(pred);

  ndt_.setInputSource(current_scan_);
  CloudPtr output(new PointCloudType);
  ndt_.align(*output, pred.matrix().cast<float>());

  SE3 pose = Mat4ToSE3(ndt_.getFinalTransformation());
  eskf_.ObserveSE3(pose, 1e-1, 1e-2);

  LOG(INFO) << "lidar loc score: " << ndt_.getTransformationProbability();

  return true;
}

void LocalizationFusion::LoadMap(const SE3 &pose) {
  int gx = floor((pose.translation().x() - 50.0) / 100);
  int gy = floor((pose.translation().y() - 50.0) / 100);
  Vec2i key(gx, gy);

  // Load the 3x3 neighborhood of map tiles around the pose
  std::set<Vec2i, less_vec<2>> surrounding_index{
      key + Vec2i(0, 0),  key + Vec2i(-1, 0), key + Vec2i(-1, -1),
      key + Vec2i(-1, 1), key + Vec2i(0, -1), key + Vec2i(0, 1),
      key + Vec2i(1, 0),  key + Vec2i(1, -1), key + Vec2i(1, 1),
  };

  // Load required tiles
  bool map_data_changed = false;
  int cnt_new_loaded = 0, cnt_unload = 0;
  for (auto &k : surrounding_index) {
    if (map_data_index_.find(k) == map_data_index_.end()) {
      // No map data for this tile
      continue;
    }

    if (map_data_.find(k) == map_data_.end()) {
      // Load this tile
      CloudPtr cloud(new PointCloudType);
      pcl::io::loadPCDFile(data_path_ + std::to_string(k[0]) + "_" +
                               std::to_string(k[1]) + ".pcd",
                           *cloud);
      map_data_.emplace(k, cloud);
      map_data_changed = true;
      cnt_new_loaded++;
    }
  }

  // Unload distant tiles (radius kept large to avoid frequent reload)
  for (auto iter = map_data_.begin(); iter != map_data_.end();) {
    if ((iter->first - key).cast<float>().norm() > 3.0) {
      // Unload this tile
      iter = map_data_.erase(iter);
      cnt_unload++;
      map_data_changed = true;
    } else {
      iter++;
    }
  }

  LOG(INFO) << "new loaded: " << cnt_new_loaded << ", unload: " << cnt_unload;
  if (map_data_changed) {
    // rebuild ndt target map
    ref_cloud_.reset(new PointCloudType);
    for (auto &mp : map_data_) {
      *ref_cloud_ += *mp.second;
    }

    LOG(INFO) << "rebuild global cloud, grids: " << map_data_.size();
    ndt_.setInputTarget(ref_cloud_);
  }

  ui_->UpdatePointCloudGlobal(map_data_);
}

void LocalizationFusion::LoadMapIndex() {
  std::ifstream fin(data_path_ + "/map_index.txt");
  while (!fin.eof()) {
    int x, y;
    fin >> x >> y;
    map_data_index_.emplace(Vec2i(x, y));
  }
  fin.close();
}

void LocalizationFusion::ProcessIMU(IMUPtr imu) { sync_->ProcessIMU(imu); }

void LocalizationFusion::ProcessPointCloud(
    sensor_msgs::PointCloud2::Ptr cloud) {
  sync_->ProcessCloud(cloud);
}

} // namespace sad