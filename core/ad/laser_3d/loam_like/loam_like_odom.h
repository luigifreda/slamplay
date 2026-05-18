#ifndef SLAM_IN_AUTO_DRIVING_LOAM_LIKE_ODOM_H
#define SLAM_IN_AUTO_DRIVING_LOAM_LIKE_ODOM_H

#include "ad/kdtree/kdtree.h"
#include "ad/laser_3d/icp_3d.h"

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include "viz/ad/pcl_map_viewer.h"

#include <deque>

namespace sad {
class FeatureExtraction;

/**
 * LOAM-like odometry method.
 * First uses feature extraction to extract edge points and planar points from a
 * point cloud, then applies different ICP methods for edge points and surface
 * points respectively.
 */
class LoamLikeOdom {
public:
  struct Options {
    Options() {}

    int min_edge_pts_ = 20;         // minimum number of edge points
    int min_surf_pts_ = 20;         // minimum number of planar points
    double kf_distance_ = 1.0;      // keyframe distance threshold
    double kf_angle_deg_ = 15;      // rotation angle threshold
    int num_kfs_in_local_map_ = 30; // number of keyframes in the local map
    bool display_realtime_cloud_ =
        true; // whether to display the real-time point cloud

    // ICP parameters
    int max_iteration_ = 5; // maximum number of iterations
    double max_plane_distance_ =
        0.05; // threshold for plane nearest-neighbor search
    double max_line_distance_ =
        0.5; // threshold for point-to-line nearest-neighbor search
    int min_effective_pts_ =
        10;             // minimum effective nearest-neighbor points threshold
    double eps_ = 1e-3; // convergence criterion

    bool use_edge_points_ = true; // whether to use edge points
    bool use_surf_points_ = true; // whether to use planar points
  };

  explicit LoamLikeOdom(Options options = Options());

  /**
   * Add a point cloud to the odometry; internally splits into edge and planar
   * points.
   * @param full_cloud
   */
  void ProcessPointCloud(FullCloudPtr full_cloud);

  void SaveMap(const std::string &path);

private:
  /// Align with the local map
  SE3 AlignWithLocalMap(CloudPtr edge, CloudPtr surf);

  /// Determine whether this is a keyframe
  bool IsKeyframe(const SE3 &current_pose);

  Options options_;

  int cnt_frame_ = 0;
  int last_kf_id_ = 0;

  CloudPtr local_map_edge_ = nullptr,
           local_map_surf_ = nullptr; // local map for the local map
  std::vector<SE3> estimated_poses_; // all estimated poses, used for trajectory
                                     // recording and next-frame prediction
  SE3 last_kf_pose_;                 // pose of the last keyframe
  std::deque<CloudPtr> edges_, surfs_; // cached edge and planar points

  CloudPtr global_map_ = nullptr; // global map for saving

  std::shared_ptr<FeatureExtraction> feature_extraction_ = nullptr;

  std::shared_ptr<PCLMapViewer> viewer_ = nullptr;
  KdTree kdtree_edge_, kdtree_surf_;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_LOAM_LIKE_ODOM_H
