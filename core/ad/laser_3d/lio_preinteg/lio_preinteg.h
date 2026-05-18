#ifndef SAD_CH8_LIO_PREINTEG_H
#define SAD_CH8_LIO_PREINTEG_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

/// Some classes directly reuse results from ch. 7
#include "ad/imu/imu_preintegration.h"
#include "ad/imu/static_imu_init.h"
#include "ad/laser_3d/message_sync/message_sync.h"
#include "ad/laser_3d/ndt_inc.h"
#include "ad/pointcloud/cloud_convert.h"

#include "ad/common/math_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

namespace sad {

/**
 * Ch. 8 LIO with IMU preintegration.
 * Same pipeline as IEKF LIO; the IEKF update is replaced by preintegration + g2o.
 */
class LioPreinteg {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  struct Options {
    Options() {}
    bool with_ui_ = true; // enable UI
    bool verbose_ = true; // print debug logs

    double bias_gyro_var_ = 1e-2;          // gyro bias random-walk std dev
    double bias_acce_var_ = 1e-2;          // accel bias random-walk std dev
    Mat3d bg_rw_info_ = Mat3d::Identity(); // gyro bias RW information matrix
    Mat3d ba_rw_info_ = Mat3d::Identity(); // accel bias RW information matrix

    double ndt_pos_noise_ = 0.1;                  // NDT position variance
    double ndt_ang_noise_ = 2.0 * math::kDEG2RAD; // NDT orientation variance
    Mat6d ndt_info_ = Mat6d::Identity();          // 6D NDT information matrix

    sad::IMUPreintegration::Options preinteg_options_; // preintegration options
    IncNdt3d::Options ndt_options_;                    // NDT options
  };

  LioPreinteg(Options options = Options());
  ~LioPreinteg() = default;

  /// init without ros
  bool Init(const std::string &config_yaml);

  /// point cloud callbacks
  void PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);

  /// IMU callback
  void IMUCallBack(IMUPtr msg_in);

  /// stop processing and quit UI
  void Finish();

private:
  bool LoadFromYAML(const std::string &yaml_file);

  /// process synchronized IMU and LiDAR data
  void ProcessMeasurements(const MeasureGroup &meas);

  /// attempt IMU initialization
  void TryInitIMU();

  /// propagate state with IMU; predictions are stored in imu_states_
  void Predict();

  /// motion-compensate the point cloud in measures_
  void Undistort();

  /// run registration and observation update once
  void Align();

  /// optimize with preintegration + NDT pose factor
  void Optimize();

  /// clamp velocity to plausible range
  void NormalizeVelocity();

  /// modules
  std::shared_ptr<MessageSync> sync_ = nullptr;
  StaticIMUInit imu_init_;

  /// point clouds data
  FullCloudPtr scan_undistort_{
      new FullPointCloudType()}; // scan after undistortion
  CloudPtr current_scan_ = nullptr;

  // optimization state
  NavStated last_nav_state_, current_nav_state_; // previous and current nav state
  Mat15d prior_info_ = Mat15d::Identity();       // marginal prior on current state
  std::shared_ptr<IMUPreintegration> preinteg_ = nullptr;

  IMUPtr last_imu_ = nullptr;

  /// NDT map / registration
  IncNdt3d ndt_;
  SE3 ndt_pose_;
  SE3 last_ndt_pose_;

  // flags
  bool imu_need_init_ = true;
  bool flg_first_scan_ = true;
  int frame_num_ = 0;

  MeasureGroup measures_; // sync IMU and lidar scan
  std::vector<NavStated> imu_states_;
  SE3 TIL_; // extrinsic: LiDAR w.r.t. IMU

  Options options_;
  std::shared_ptr<ui::AutonomousDrivingViz> ui_ = nullptr;
};

} // namespace sad

#endif // FASTER_LIO_LASER_MAPPING_H