#ifndef SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H
#define SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H

#include <opencv2/core.hpp>

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/lidar_utils.h"

namespace sad {

class LikelihoodField {
public:
  /// 2D field template, generated when setting target scan or map
  struct ModelPoint {
    ModelPoint(int dx, int dy, float res) : dx_(dx), dy_(dy), residual_(res) {}
    int dx_ = 0;
    int dy_ = 0;
    float residual_ = 0;
  };

  LikelihoodField() { BuildModel(); }

  /// Set a 2D target scan
  void SetTargetScan(Scan2d::Ptr scan);

  /// Set the source scan to be aligned
  void SetSourceScan(Scan2d::Ptr scan);

  /// Generate a likelihood field map from an occupancy grid map
  void SetFieldImageFromOccuMap(const cv::Mat &occu_map);

  /// Align using Gauss-Newton method
  bool AlignGaussNewton(SE2 &init_pose);

  /**
   * Align using g2o
   * @param init_pose initial pose. NOTE: when using submaps, provide the pose
   * relative to that submap; the result will also be relative to that submap
   * @return
   */
  bool AlignG2O(SE2 &init_pose);

  /// Get the field function, converted to RGB image
  cv::Mat GetFieldImage();

  bool HasOutsidePoints() const { return has_outside_pts_; }

  void SetPose(const SE2 &pose) { pose_ = pose; }

private:
  void BuildModel();

  SE2 pose_; // T_W_S
  Scan2d::Ptr target_ = nullptr;
  Scan2d::Ptr source_ = nullptr;

  std::vector<ModelPoint> model_; // 2D template
  cv::Mat field_;                 // field function
  bool has_outside_pts_ = false;  // whether any points fell outside this field

  // Parameter configuration
  inline static const float resolution_ = 20; // pixels per meter
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H
