#ifndef SLAM_IN_AUTO_DRIVING_OCCUPANCY_MAP_H
#define SLAM_IN_AUTO_DRIVING_OCCUPANCY_MAP_H

#include <opencv2/core.hpp>

#include "ad/common/eigen_types.h"
#include "ad/laser_2d/frame.h"

namespace sad {

/// Occupancy grid map, corresponds to section 6.3 in the book
class OccupancyMap {
public:
  /// Grid template, precomputed
  struct Model2DPoint {
    int dx_ = 0;
    int dy_ = 0;
    double angle_ = 0; // in rad
    float range_ = 0;  // in meters
  };

  enum class GridMethod {
    MODEL_POINTS, // template-based algorithm
    BRESENHAM,    // direct rasterization algorithm
  };

  OccupancyMap();

  /// Add a frame to this occupancy grid map
  void AddLidarFrame(std::shared_ptr<Frame> frame,
                     GridMethod method = GridMethod::BRESENHAM);

  /// Get the raw occupancy grid map
  cv::Mat GetOccupancyGrid() const { return occupancy_grid_; }

  /// Get the occupancy grid in black/white/gray form for visualization
  cv::Mat GetOccupancyGridBlackWhite() const;

  /// Set center pose
  void SetPose(const SE2 &pose) { pose_ = pose; }

  bool HasOutsidePoints() const { return has_outside_pts_; }

  /// Get resolution
  double Resolution() const { return resolution_; }

  /// Mark a point as occupied or free
  void SetPoint(const Vec2i &pt, bool occupy);

private:
  /// Build the filling template
  void BuildModel();

  /// Convert from world coordinates to image coordinates
  template <class T>
  inline Vec2i World2Image(const Eigen::Matrix<T, 2, 1> &pt) {
    Vec2d pt_map = (pose_.inverse() * pt) * resolution_ + center_image_;
    int x = int(pt_map[0]);
    int y = int(pt_map[1]);
    return Vec2i(x, y);
  }

  /// Find the range value at a given angle
  double FindRangeInAngle(double angle, Scan2d::Ptr scan);

  /**
   * Bresenham line filling: given start and end points, fill the area in
   * between as free
   * @param p1
   * @param p2
   */
  void BresenhamFilling(const Vec2i &p1, const Vec2i &p2);

  cv::Mat occupancy_grid_; // 8-bit occupancy grid image

  SE2 pose_; // T_W_S
  Vec2d center_image_ = Vec2d(image_size_ / 2, image_size_ / 2);

  bool has_outside_pts_ =
      false; // whether any points fell outside during rasterization

  // Template
  std::vector<Model2DPoint> model_; // template for filling the occupancy grid,
                                    // all points in world frame

  // Parameters
  inline static constexpr double closest_th_ = 0.2; // close range threshold
  inline static constexpr double endpoint_close_th_ =
      0.1; // endpoint obstacle close range threshold
  inline static constexpr double resolution_ = 20.0; // pixels per meter
  inline static constexpr float inv_resolution_ =
      0.05; // meters per pixel (grid resolution)
  inline static constexpr int image_size_ = 1000; // image size
  inline static constexpr int model_size_ = 400;  // template pixel size
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_OCCUPANCY_MAP_H
