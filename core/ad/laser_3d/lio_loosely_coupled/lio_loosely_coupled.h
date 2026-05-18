#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

#include "ad/laser_3d/message_sync/message_sync.h"
#include "ad/laser_3d/ndt_lo_incremental.h"
#include "ad/pointcloud/cloud_convert.h"

#include "ad/imu/static_imu_init.h"
#include "ad/kf/eskf.hpp"

#include "viz/ad/autonomous_driving_viz.h"

namespace sad {

/**
 * Loosely-coupled LIO program (Section 7.5)
 * Implemented using the EKF from Chapter 3 and the incremental NDT odometry
 * from Section 7.3 Required parameters are read from a YAML file beforehand
 *
 * Since this is the first program with both IMU and LiDAR, the framework has
 * significant changes The subsequent tightly-coupled LIO will continue to use
 * this framework
 */
class LIOLooselyCoupled {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  struct Options {
    Options() {}
    bool save_motion_undistortion_pcd_ =
        false; // Whether to save point clouds before and after undistortion
    bool with_ui_ = true; // Whether to enable UI
  };

  LIOLooselyCoupled(Options options);
  ~LIOLooselyCoupled() = default;

  /// Initialize from config file
  bool Init(const std::string &config_yaml);

  /// Point cloud callback
  void PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);

  /// IMU callback
  void IMUCallBack(IMUPtr msg_in);

  /// Finish program, exit UI
  void Finish();

private:
  /// Process synchronized IMU and LiDAR data
  void ProcessMeasurements(const MeasureGroup &meas);

  bool LoadFromYAML(const std::string &yaml);

  /// Try to initialize IMU
  void TryInitIMU();

  /// Predict state using IMU
  /// Predicted states during this period are stored in imu_states_
  void Predict();

  /// Undistort the point cloud in measures_
  void Undistort();

  /// Perform one registration and observation
  void Align();

private:
  /// modules
  std::shared_ptr<MessageSync> sync_ = nullptr; // Message synchronizer
  StaticIMUInit imu_init_;                      // IMU static initialization
  std::shared_ptr<sad::NDTLOIncremental> inc_ndt_lo_ = nullptr;

  /// point clouds data
  FullCloudPtr scan_undistort_{
      new FullPointCloudType()}; // scan after undistortion
  SE3 pose_of_lo_;

  Options options_;

  // flags
  bool imu_need_init_ = true;  // Whether IMU initial bias estimation is needed
  bool flg_first_scan_ = true; // Whether this is the first LiDAR scan
  int frame_num_ = 0;          // Frame counter

  // EKF data
  MeasureGroup measures_;             // Synchronized IMU and point cloud data
  std::vector<NavStated> imu_states_; // States during ESKF prediction
  ESKFD eskf_;                        // ESKF
  SE3 TIL_;                           // LiDAR-IMU extrinsic parameters

  std::shared_ptr<ui::AutonomousDrivingViz> ui_ = nullptr;
};

} // namespace sad

#endif // FASTER_LIO_LASER_MAPPING_H