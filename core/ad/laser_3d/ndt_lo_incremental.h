#ifndef SLAM_IN_AUTO_DRIVING_INCREMENTAL_NDT_LO_H
#define SLAM_IN_AUTO_DRIVING_INCREMENTAL_NDT_LO_H

#include "ad/common/eigen_types.h"
#include "ad/laser_3d/ndt_inc.h"
#include "ad/pointcloud/point_types.h"

#include "viz/ad/pcl_map_viewer.h"

namespace sad {

/**
 * Lidar Odometry using incremental NDT method
 * Uses historical keyframes as local map for NDT localization
 */
class NDTLOIncremental {
public:
  struct Options {
    Options() {}
    double kf_distance_ = 0.5; // keyframe distance
    double kf_angle_deg_ = 30; // rotation angle
    bool display_realtime_cloud_ =
        true; // whether to display real-time point cloud
    IncNdt3d::Options ndt3d_options_; // NDT3D configuration
  };

  NDTLOIncremental(Options options = Options()) : options_(options) {
    if (options_.display_realtime_cloud_) {
      viewer_ = std::make_shared<PCLMapViewer>(0.5);
    }

    ndt_ = IncNdt3d(options_.ndt3d_options_);
  }

  /**
   * Add a point cloud to the LO
   * @param scan  current frame point cloud
   * @param pose  estimated pose
   */
  void AddCloud(CloudPtr scan, SE3 &pose, bool use_guess = false);

  /// save map (in viewer)
  void SaveMap(const std::string &map_path);

private:
  /// determine whether this is a keyframe
  bool IsKeyframe(const SE3 &current_pose);

private:
  Options options_;
  bool first_frame_ = true;
  std::vector<SE3> estimated_poses_; // all estimated poses, for recording
                                     // trajectory and predicting the next frame
  SE3 last_kf_pose_;                 // pose of the previous keyframe
  int cnt_frame_ = 0;

  IncNdt3d ndt_;
  std::shared_ptr<PCLMapViewer> viewer_ = nullptr;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_INCREMENTAL_NDT_LO_H
