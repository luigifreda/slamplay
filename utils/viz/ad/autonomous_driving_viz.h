#ifndef SAD_UI_PANGOLIN_WINDOW_H
#define SAD_UI_PANGOLIN_WINDOW_H

#include "ad/common/eigen_types.h"
#include "ad/nav/gnss.h"
#include "ad/nav/nav_state.h"
#include "ad/pointcloud/point_types.h"

#include <map>
#include <memory>

namespace sad::ui {

class AutonomousDrivingVizImpl;

/**
 * 3D Visualization for Autonomous Driving System
 * This class is used to visualize the autonomous driving system in 3D.
 * @note This class itself should not directly involve any OpenGL or Pangolin
 * operations; those should be placed in `AutonomousDrivingVizImpl`.
 */
class AutonomousDrivingViz {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  AutonomousDrivingViz();
  ~AutonomousDrivingViz();

  /// @brief Initialize the window and start the render thread in the
  /// background.
  /// @note Keep initialization unrelated to OpenGL/Pangolin in this function
  /// body;
  ///       keep OpenGL/Pangolin-related content in
  ///       `AutonomousDrivingVizImpl::Init`.
  bool Init();

  /// Update the LiDAR map point cloud; called by fusion when the localization
  /// map changes
  void
  UpdatePointCloudGlobal(const std::map<Vec2i, CloudPtr, less_vec<2>> &cloud);

  /// Update the Kalman filter state
  void UpdateNavState(const NavStated &state);

  /// Update one scan and its corresponding pose
  void UpdateScan(CloudPtr cloud, const SE3 &pose);

  /// Update the GPS localization result
  void UpdateGps(const GNSS &gps);

  /// Wait for the display thread to finish and release resources
  void Quit();

  /// Whether the user has already exited the UI
  bool ShouldQuit();

  /// Set the extrinsic transform from IMU to LiDAR
  void SetTImuLidar(const SE3 &T_imu_lidar);

  /// Set how many scans should be retained
  void SetCurrentScanSize(int current_scan_size);

private:
  std::shared_ptr<AutonomousDrivingVizImpl> impl_ = nullptr;
};
} // namespace sad::ui

#endif
