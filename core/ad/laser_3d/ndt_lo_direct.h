#ifndef SLAM_IN_AUTO_DRIVING_DIRECT_NDT_LO_H
#define SLAM_IN_AUTO_DRIVING_DIRECT_NDT_LO_H

#include <deque>
#include <pcl/registration/ndt.h>

#include "ad/common/eigen_types.h"
#include "ad/laser_3d/ndt_3d.h"
#include "ad/pointcloud/point_types.h"
#include "viz/ad/pcl_map_viewer.h"

namespace sad {

/**
 * Lidar Odometry using direct NDT method
 * Uses historical keyframes as local map for NDT localization
 */
class NDTLODirect {
public:
  struct Options {
    Options() {}
    double kf_distance_ = 0.5;      // keyframe distance
    double kf_angle_deg_ = 30;      // rotation angle
    int num_kfs_in_local_map_ = 30; // number of keyframes in local map
    bool use_pcl_ndt_ = true;       // use this chapter's NDT or PCL's NDT
    bool display_realtime_cloud_ =
        true; // whether to display real-time point cloud

    Ndt3d::Options ndt3d_options_; // NDT3D configuration
  };

  NDTLODirect(Options options = Options()) : options_(options) {
    if (options_.display_realtime_cloud_) {
      viewer_ = std::make_shared<PCLMapViewer>(0.5);
    }

    ndt_ = Ndt3d(options_.ndt3d_options_);

    ndt_pcl_.setResolution(1.0);
    ndt_pcl_.setStepSize(0.1);
    ndt_pcl_.setTransformationEpsilon(0.01);
  }

  /**
   * Add a point cloud to the LO
   * @param scan  current frame point cloud
   * @param pose  estimated pose
   */
  void AddCloud(CloudPtr scan, SE3 &pose);

  /// save map (in viewer)
  void SaveMap(const std::string &map_path);

private:
  /// align with the local map
  SE3 AlignWithLocalMap(CloudPtr scan);

  /// determine whether this is a keyframe
  bool IsKeyframe(const SE3 &current_pose);

private:
  Options options_;
  CloudPtr local_map_ = nullptr;
  std::deque<CloudPtr> scans_in_local_map_;
  std::vector<SE3> estimated_poses_; // all estimated poses, for recording
                                     // trajectory and predicting the next frame
  SE3 last_kf_pose_;                 // pose of the previous keyframe

  pcl::NormalDistributionsTransform<PointType, PointType> ndt_pcl_;
  Ndt3d ndt_;

  std::shared_ptr<PCLMapViewer> viewer_ = nullptr;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_DIRECT_NDT_LO_H
