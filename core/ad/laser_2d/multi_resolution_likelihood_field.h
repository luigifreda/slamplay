#ifndef SLAM_IN_AUTO_DRIVING_MULTI_RESOLUTION_LIKELIHOOD_FILED_H
#define SLAM_IN_AUTO_DRIVING_MULTI_RESOLUTION_LIKELIHOOD_FILED_H

#include <opencv2/core.hpp>

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/lidar_utils.h"

namespace sad {

/// Multi-resolution likelihood field alignment method
class MRLikelihoodField {
public:
  /// 2D field template, generated when setting target scan or map
  struct ModelPoint {
    ModelPoint(int dx, int dy, float res) : dx_(dx), dy_(dy), residual_(res) {}
    int dx_ = 0;
    int dy_ = 0;
    float residual_ = 0;
  };

  MRLikelihoodField() { BuildModel(); }

  /// Generate a likelihood field map from an occupancy grid map
  void SetFieldImageFromOccuMap(const cv::Mat &occu_map);

  /// Align using g2o, internally aligns at different pyramid levels
  bool AlignG2O(SE2 &init_pose);

  /// Get the field function, converted to RGB images
  std::vector<cv::Mat> GetFieldImage();

  /// Set center (usually the submap center)
  void SetPose(const SE2 &pose) { pose_ = pose; }

  /// Set the source for matching
  void SetSourceScan(Scan2d::Ptr scan) { source_ = scan; }

  float Resolution(int level = 0) const { return resolution_[level]; }

  int Levels() const { return levels_; }

private:
  /**
   * Align within a certain pyramid level
   * @param level
   * @param init_pose
   * @return
   */
  bool AlignInLevel(int level, SE2 &init_pose);

  void BuildModel();

  SE2 pose_;

  Scan2d::Ptr source_ = nullptr;

  std::vector<ModelPoint> model_; // 2D template
  std::vector<cv::Mat> field_;    // field function

  std::vector<int> num_inliers_;
  std::vector<double> inlier_ratio_;

  // Parameter configuration
  inline static const int levels_ = 4; // number of pyramid levels
  inline static const std::vector<float> resolution_ = {2.5, 5, 10,
                                                        20}; // pixels per meter
  inline static const std::vector<float> ratios_ = {
      0.125, 0.25, 0.5, 1.0}; // scale ratio relative to the occupancy grid
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_MULTI_RESOLUTION_LIKELIHOOD_FILED_H
