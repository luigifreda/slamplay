#ifndef SAD_CH8_LASER_MAPPING_H
#define SAD_CH8_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

/// Some classes directly reuse results from ch. 7
#include "ad/imu/static_imu_init.h"
#include "ad/laser_3d/lio_iekf/iekf.hpp"
#include "ad/laser_3d/message_sync/message_sync.h"
#include "ad/laser_3d/ndt_inc.h"
#include "ad/pointcloud/cloud_convert.h"

#include "viz/ad/autonomous_driving_viz.h"

namespace sad {

class LioIEKF {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  struct Options {
    Options() {}
    bool save_motion_undistortion_pcd_ =
        false;            // save point clouds before/after undistortion
    bool with_ui_ = true; // enable UI
  };

  LioIEKF(Options options = Options());
  ~LioIEKF() = default;

  /// init without ros
  bool Init(const std::string &config_yaml);

  /// point cloud callbacks
  void PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);

  /// IMU callback
  void IMUCallBack(IMUPtr msg_in);

  /// stop processing and quit UI
  void Finish();

  /// current navigation state
  NavStated GetCurrentState() const { return ieskf_.GetNominalState(); }

  /// current scan
  CloudPtr GetCurrentScan() const { return current_scan_; }

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

  /// modules
  std::shared_ptr<MessageSync> sync_ = nullptr;
  StaticIMUInit imu_init_;

  /// point clouds data
  FullCloudPtr scan_undistort_{
      new FullPointCloudType()}; // scan after undistortion
  CloudPtr current_scan_ = nullptr;

  /// NDT map / registration
  IncNdt3d ndt_;
  SE3 last_pose_;

  // flags
  bool imu_need_init_ = true;
  bool flg_first_scan_ = true;
  int frame_num_ = 0;

  ///////////////////////// EKF inputs and output
  //////////////////////////////////////////////////////////
  MeasureGroup measures_; // sync IMU and lidar scan
  std::vector<NavStated> imu_states_;
  IESKFD ieskf_; // IESKF
  SE3 TIL_;      // extrinsic: LiDAR w.r.t. IMU

  Options options_;
  std::shared_ptr<ui::AutonomousDrivingViz> ui_ = nullptr;
};

} // namespace sad

#endif // FASTER_LIO_LASER_MAPPING_H