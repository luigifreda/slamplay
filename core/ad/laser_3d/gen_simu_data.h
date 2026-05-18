#ifndef SLAM_IN_AUTO_DRIVING_GEN_SIMU_DATA_H
#define SLAM_IN_AUTO_DRIVING_GEN_SIMU_DATA_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

namespace sad {

/**
 * Generate simulation data needed for this chapter
 * Simulation data provides ground truth poses and guarantees point-to-point
 * correspondences, allowing us to verify algorithm correctness
 *
 * This program simulates a simple cuboid, applies a random 6-DOF
 * transformation, and outputs source and target point clouds
 */
class GenSimuData {
public:
  struct Options {
    Options() {}
    int num_points_ = 2000; // number of points
    // box parameters
    double width_ = 5.0;   // width, y-direction
    double length_ = 10.0; // length, x-direction
    double height_ = 1.0;  // height, z-direction
    // pose parameters
    double pose_rot_sigma_ = 0.05;  // rotation sigma
    double pose_trans_sigma_ = 0.3; // translation sigma
  };

  GenSimuData(Options options = Options()) : options_(options) {}

  /// generate target and source point clouds
  void Gen();

  CloudPtr GetTarget() const { return target_; }
  CloudPtr GetSource() const { return source_; }
  SE3 GetPose() const { return gt_pose_; }

private:
  void GenTarget();

  CloudPtr target_ = nullptr, source_ = nullptr;
  Options options_;
  SE3 gt_pose_; // ground truth pose, from target to source
};
} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_GEN_SIMU_DATA_H
