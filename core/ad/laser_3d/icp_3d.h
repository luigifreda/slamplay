#ifndef SLAM_IN_AUTO_DRIVING_ICP_3D_H
#define SLAM_IN_AUTO_DRIVING_ICP_3D_H

#include "ad/kdtree/kdtree.h"

namespace sad {

/**
 * 3D ICP
 * Call SetTarget first, then SetSource, then call an Align method to obtain the pose
 *
 * ICP solves for R, t to align the source point cloud to the target point cloud
 * If p is a point in the source cloud, then R*p+t gives the corresponding point in the target cloud
 *
 * Uses the KD-tree from Chapter 5 for nearest neighbor queries
 */
class Icp3d {
public:
  struct Options {
    int max_iteration_ = 20;               // max number of iterations
    double max_nn_distance_ = 1.0;         // threshold for point-to-point nearest neighbor search
    double max_plane_distance_ = 0.05;     // threshold for plane nearest neighbor search
    double max_line_distance_ = 0.5;       // threshold for point-to-line nearest neighbor search
    int min_effective_pts_ = 10;           // min effective points threshold
    double eps_ = 1e-2;                    // convergence criterion
    bool use_initial_translation_ = false; // whether to use translation from initial pose
  };

  Icp3d() {}
  Icp3d(Options options) : options_(options) {}

  /// set the target scan
  void SetTarget(CloudPtr target) {
    target_ = target;
    BuildTargetKdTree();

    // compute point cloud centroid
    target_center_ =
        std::accumulate(target->points.begin(), target_->points.end(),
                        Vec3d::Zero().eval(),
                        [](const Vec3d &c, const PointType &pt) -> Vec3d {
                          return c + ToVec3d(pt);
                        }) /
        target_->size();
    LOG(INFO) << "target center: " << target_center_.transpose();
  }

  /// set the source scan to be aligned
  void SetSource(CloudPtr source) {
    source_ = source;
    source_center_ =
        std::accumulate(source_->points.begin(), source_->points.end(),
                        Vec3d::Zero().eval(),
                        [](const Vec3d &c, const PointType &pt) -> Vec3d {
                          return c + ToVec3d(pt);
                        }) /
        source_->size();
    LOG(INFO) << "source center: " << source_center_.transpose();
  }

  void SetGroundTruth(const SE3 &gt_pose) {
    gt_pose_ = gt_pose;
    gt_set_ = true;
  }

  /// point-to-point alignment using Gauss-Newton method
  bool AlignP2P(SE3 &init_pose);

  /// point-to-line ICP based on Gauss-Newton
  bool AlignP2Line(SE3 &init_pose);

  /// point-to-plane ICP based on Gauss-Newton
  bool AlignP2Plane(SE3 &init_pose);

private:
  // build KD-tree for target point cloud
  void BuildTargetKdTree();

  std::shared_ptr<KdTree> kdtree_ = nullptr; // KD-tree from Chapter 5

  CloudPtr target_ = nullptr;
  CloudPtr source_ = nullptr;

  Vec3d target_center_ = Vec3d::Zero();
  Vec3d source_center_ = Vec3d::Zero();

  bool gt_set_ = false; // whether ground truth is set
  SE3 gt_pose_;

  Options options_;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_ICP_3D_H
