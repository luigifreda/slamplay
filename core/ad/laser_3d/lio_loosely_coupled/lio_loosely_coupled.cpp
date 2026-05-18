#include <execution>
#include <yaml-cpp/yaml.h>

#include "ad/laser_3d/lio_loosely_coupled/lio_loosely_coupled.h"
#include "ad/pointcloud/lidar_utils.h"
#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/timer/timer.h"

#include "macros.h"
#include <filesystem>

namespace {
std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
} // namespace

namespace sad {
std::string resultsLioLooselyCoupledPath =
    resultsDir + "/ad/laser_3d/lio_loosely_coupled";

LIOLooselyCoupled::LIOLooselyCoupled(Options options) : options_(options) {
  StaticIMUInit::Options imu_init_options;
  imu_init_options.use_speed_for_static_checking_ =
      false; // This section's data does not need wheel odometer
  imu_init_ = StaticIMUInit(imu_init_options);

  if (!std::filesystem::exists(resultsLioLooselyCoupledPath)) {
    std::filesystem::create_directories(resultsLioLooselyCoupledPath);
  }
}

bool LIOLooselyCoupled::Init(const std::string &config_yaml) {
  /// Initialize own parameters
  if (!LoadFromYAML(config_yaml)) {
    return false;
  }

  /// Initialize NDT LO parameters
  sad::NDTLOIncremental::Options indt_options;
  indt_options.display_realtime_cloud_ =
      false; // This program has its own UI, no need for PCL's viewer
  inc_ndt_lo_ = std::make_shared<sad::NDTLOIncremental>(indt_options);

  /// Initialize UI
  if (options_.with_ui_) {
    ui_ = std::make_shared<ui::AutonomousDrivingViz>();
    ui_->Init();
  }

  return true;
}

bool LIOLooselyCoupled::LoadFromYAML(const std::string &yaml_file) {
  // get params from yaml
  sync_ = std::make_shared<MessageSync>(
      [this](const MeasureGroup &m) { ProcessMeasurements(m); });
  sync_->Init(yaml_file);

  /// Own parameters are mainly the LiDAR-IMU extrinsics
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

void LIOLooselyCoupled::ProcessMeasurements(const MeasureGroup &meas) {
  LOG(INFO) << "call meas, imu: " << meas.imu_.size()
            << ", lidar pts: " << meas.lidar_->size();
  measures_ = meas;

  if (imu_need_init_) {
    // Initialize IMU system
    TryInitIMU();
    return;
  }

  // Use IMU data for state prediction
  Predict();

  // Undistort point cloud
  Undistort();

  // Registration
  Align();
}

void LIOLooselyCoupled::Predict() {
  imu_states_.clear();
  imu_states_.emplace_back(eskf_.GetNominalState());

  /// Predict IMU states
  for (auto &imu : measures_.imu_) {
    eskf_.Predict(*imu);
    imu_states_.emplace_back(eskf_.GetNominalState());
  }
}

void LIOLooselyCoupled::TryInitIMU() {
  for (auto imu : measures_.imu_) {
    imu_init_.AddIMU(*imu);
  }

  if (imu_init_.InitSuccess()) {
    // Read initial biases, set up ESKF
    sad::ESKFD::Options options;
    // Noise estimated by initializer
    options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
    options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
    eskf_.SetInitialConditions(options, imu_init_.GetInitBg(),
                               imu_init_.GetInitBa(), imu_init_.GetGravity());
    imu_need_init_ = false;

    LOG(INFO) << "IMU initialization successful";
  }
}

void LIOLooselyCoupled::Undistort() {
  auto cloud = measures_.lidar_;
  auto imu_state = eskf_.GetNominalState(); // State at the last moment
  SE3 T_end = SE3(imu_state.R_, imu_state.p_);

  if (options_.save_motion_undistortion_pcd_) {
    sad::SaveCloudToFile(resultsLioLooselyCoupledPath + "/before_undist.pcd",
                         *cloud);
  }

  /// Transform all points to the state at the last moment
  std::for_each(std::execution::par_unseq, cloud->points.begin(),
                cloud->points.end(), [&](auto &pt) {
                  SE3 Ti = T_end;
                  NavStated match;

                  // Look up time based on pt.time; pt.time is the difference
                  // between the point's hit time and the LiDAR start time, in
                  // milliseconds
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
    sad::SaveCloudToFile(resultsLioLooselyCoupledPath + "/after_undist.pcd",
                         *cloud);
  }
}

void LIOLooselyCoupled::Align() {
  FullCloudPtr scan_undistort_trans(new FullPointCloudType);
  pcl::transformPointCloud(*scan_undistort_, *scan_undistort_trans,
                           TIL_.matrix());
  scan_undistort_ = scan_undistort_trans;

  auto current_scan = ConvertToCloud<FullPointType>(scan_undistort_);

  // Voxel filter
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(0.5, 0.5, 0.5);
  voxel.setInputCloud(current_scan);

  CloudPtr current_scan_filter(new PointCloudType);
  voxel.filter(*current_scan_filter);

  /// Handle the first LiDAR frame
  if (flg_first_scan_) {
    SE3 pose;
    inc_ndt_lo_->AddCloud(current_scan_filter, pose);
    flg_first_scan_ = false;
    return;
  }

  /// Get predicted pose from EKF, pass to LO, obtain LO pose, then fuse into
  /// EKF
  SE3 pose_predict = eskf_.GetNominalSE3();
  inc_ndt_lo_->AddCloud(current_scan_filter, pose_predict, true);
  pose_of_lo_ = pose_predict;
  eskf_.ObserveSE3(pose_of_lo_, 1e-2, 1e-2);

  if (options_.with_ui_) {
    // Send to UI
    ui_->UpdateScan(
        current_scan,
        eskf_.GetNominalSE3()); // Convert to LiDAR pose and pass to UI
    ui_->UpdateNavState(eskf_.GetNominalState());
  }
  frame_num_++;
}

void LIOLooselyCoupled::PCLCallBack(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  sync_->ProcessCloud(msg);
}

void LIOLooselyCoupled::LivoxPCLCallBack(
    const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  sync_->ProcessCloud(msg);
}

void LIOLooselyCoupled::IMUCallBack(IMUPtr msg_in) {
  sync_->ProcessIMU(msg_in);
}

void LIOLooselyCoupled::Finish() {
  if (options_.with_ui_) {
    while (ui_->ShouldQuit() == false) {
      usleep(1e5);
    }

    ui_->Quit();
  }
  LOG(INFO) << "finish done";
}

} // namespace sad