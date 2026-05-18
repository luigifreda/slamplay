#include <execution>
#include <fstream>
#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "ad/g2o/g2o_types.h"

#include "ad/pointcloud/lidar_utils.h"
#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/timer/timer.h"

#include "lio_iekf.h"

#include "macros.h"
#include <filesystem>

namespace {
std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
} // namespace

namespace sad {

std::string resultsLioIekfPath = resultsDir + "/ad/laser_3d/lio_iekf";

LioIEKF::LioIEKF(Options options) : options_(options) {
  StaticIMUInit::Options imu_init_options;
  imu_init_options.use_speed_for_static_checking_ =
      false; // wheel odometry not used for this dataset
  imu_init_ = StaticIMUInit(imu_init_options);

  if (!std::filesystem::exists(resultsLioIekfPath)) {
    std::filesystem::create_directories(resultsLioIekfPath);
  }
}

bool LioIEKF::Init(const std::string &config_yaml) {
  if (!LoadFromYAML(config_yaml)) {
    LOG(INFO) << "init failed.";
    return false;
  }

  if (options_.with_ui_) {
    ui_ = std::make_shared<ui::AutonomousDrivingViz>();
    ui_->Init();
  }

  return true;
}

void LioIEKF::ProcessMeasurements(const MeasureGroup &meas) {
  LOG(INFO) << "call meas, imu: " << meas.imu_.size()
            << ", lidar pts: " << meas.lidar_->size();
  measures_ = meas;

  if (imu_need_init_) {
    // initialize IMU
    TryInitIMU();
    return;
  }

  // propagate state with IMU
  Predict();

  // motion-compensate point cloud
  Undistort();

  // register scan
  Align();
}

bool LioIEKF::LoadFromYAML(const std::string &yaml_file) {
  // get params from yaml
  sync_ = std::make_shared<MessageSync>(
      [this](const MeasureGroup &m) { ProcessMeasurements(m); });
  sync_->Init(yaml_file);

  /// load extrinsic between LiDAR and IMU
  auto yaml = YAML::LoadFile(yaml_file);
  std::vector<double> ext_t =
      yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
  std::vector<double> ext_r =
      yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

  Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
  Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
  TIL_ = SE3(lidar_R_wrt_IMU, lidar_T_wrt_IMU);
  return true;
}

void LioIEKF::Align() {
  FullCloudPtr scan_undistort_trans(new FullPointCloudType);
  pcl::transformPointCloud(*scan_undistort_, *scan_undistort_trans,
                           TIL_.matrix().cast<float>());
  scan_undistort_ = scan_undistort_trans;

  current_scan_ = ConvertToCloud<FullPointType>(scan_undistort_);

  // voxel downsample
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(0.5, 0.5, 0.5);
  voxel.setInputCloud(current_scan_);

  CloudPtr current_scan_filter(new PointCloudType);
  voxel.filter(*current_scan_filter);

  /// the first scan
  if (flg_first_scan_) {
    ndt_.AddCloud(current_scan_);
    flg_first_scan_ = false;

    return;
  }

  // subsequent scans: update pose with NDT
  LOG(INFO) << "=== frame " << frame_num_;

  ndt_.SetSource(current_scan_filter);
  ieskf_.UpdateUsingCustomObserve(
      [this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
        ndt_.ComputeResidualAndJacobians(input_pose, HTVH, HTVr);
      });

  auto current_nav_state = ieskf_.GetNominalState();

  // add scan to map if motion exceeds threshold
  SE3 current_pose = ieskf_.GetNominalSE3();
  SE3 delta_pose = last_pose_.inverse() * current_pose;

  if (delta_pose.translation().norm() > 1.0 ||
      delta_pose.so3().log().norm() > math::deg2rad(10.0)) {
    // merge scan into NDT map
    CloudPtr current_scan_world(new PointCloudType);
    pcl::transformPointCloud(*current_scan_filter, *current_scan_world,
                             current_pose.matrix());
    ndt_.AddCloud(current_scan_world);
    last_pose_ = current_pose;
  }

  // update UI
  if (ui_) {
    ui_->UpdateScan(current_scan_,
                    current_nav_state.GetSE3()); // pass pose in LiDAR frame
    ui_->UpdateNavState(current_nav_state);
  }

  frame_num_++;
  return;
}

void LioIEKF::TryInitIMU() {
  for (auto imu : measures_.imu_) {
    imu_init_.AddIMU(*imu);
  }

  if (imu_init_.InitSuccess()) {
    // read initial biases and configure IEKF
    sad::IESKFD::Options options;
    // noise std devs from static IMU init
    options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
    options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
    ieskf_.SetInitialConditions(options, imu_init_.GetInitBg(),
                                imu_init_.GetInitBa(), imu_init_.GetGravity());
    imu_need_init_ = false;

    LOG(INFO) << "IMU initialization succeeded";
  }
}

void LioIEKF::Undistort() {
  auto cloud = measures_.lidar_;
  auto imu_state = ieskf_.GetNominalState(); // state at scan end time
  SE3 T_end = SE3(imu_state.R_, imu_state.p_);

  if (options_.save_motion_undistortion_pcd_) {
    sad::SaveCloudToFile(resultsLioIekfPath + "/before_undist.pcd", *cloud);
  }

  /// transform all points to the state at scan end time
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

  if (options_.save_motion_undistortion_pcd_) {
    sad::SaveCloudToFile(resultsLioIekfPath + "/after_undist.pcd", *cloud);
  }
}

void LioIEKF::Predict() {
  imu_states_.clear();
  imu_states_.emplace_back(ieskf_.GetNominalState());

  /// propagate IMU state over the measurement interval
  for (auto &imu : measures_.imu_) {
    ieskf_.Predict(*imu);
    imu_states_.emplace_back(ieskf_.GetNominalState());
  }
}

void LioIEKF::PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  sync_->ProcessCloud(msg);
}

void LioIEKF::LivoxPCLCallBack(
    const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  sync_->ProcessCloud(msg);
}

void LioIEKF::IMUCallBack(IMUPtr msg_in) { sync_->ProcessIMU(msg_in); }

void LioIEKF::Finish() {
  if (ui_) {
    ui_->Quit();
  }
  LOG(INFO) << "finish done";
}

} // namespace sad