#ifndef SLAM_IN_AUTO_DRIVING_ICP_2D_H
#define SLAM_IN_AUTO_DRIVING_ICP_2D_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/lidar_utils.h"

#include <pcl/search/kdtree.h>

namespace sad {

/**
 * Various types of ICP implementations discussed in Chapter 6
 * Usage: first call SetTarget (builds the KD tree for the target cloud), then
 * SetSource, then call an Align* method
 */
class Icp2d {
public:
  using Point2d = pcl::PointXY;
  using Cloud2d = pcl::PointCloud<Point2d>;
  Icp2d() {}

  /// Set the target scan
  void SetTarget(Scan2d::Ptr target) {
    target_scan_ = target;
    BuildTargetKdTree();
  }

  /// Set the source scan to be aligned
  void SetSource(Scan2d::Ptr source) { source_scan_ = source; }

  /// Align using Gauss-Newton method
  bool AlignGaussNewton(SE2 &init_pose);

  /// Align using Gauss-Newton method, Point-to-Plane
  bool AlignGaussNewtonPoint2Plane(SE2 &init_pose);

private:
  // Build Kdtree for the target point cloud
  void BuildTargetKdTree();

  pcl::search::KdTree<Point2d> kdtree_;
  Cloud2d::Ptr target_cloud_; // target cloud in PCL format

  Scan2d::Ptr target_scan_ = nullptr;
  Scan2d::Ptr source_scan_ = nullptr;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_ICP_2D_H
