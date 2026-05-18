#ifndef FUSION_UI_PANGOLIN_WINDOW_IMPL_H
#define FUSION_UI_PANGOLIN_WINDOW_IMPL_H

// Include pangolin before PCL to suppress the HAVE_OPENNI compilation warning
#include <pangolin/pangolin.h>

#include "ad/pointcloud/point_types.h"
#include "autonomous_driving_viz.h"
#include "ui_car.h"
#include "ui_cloud.h"
#include "ui_trajectory.h"

#include <atomic>
#include <mutex>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <thread>

namespace sad::ui {

struct UiFrame;

/**
 */
class AutonomousDrivingVizImpl {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  AutonomousDrivingVizImpl() = default;
  ~AutonomousDrivingVizImpl() = default;

  AutonomousDrivingVizImpl(const AutonomousDrivingVizImpl &) = delete;
  AutonomousDrivingVizImpl &
  operator=(const AutonomousDrivingVizImpl &) = delete;
  AutonomousDrivingVizImpl(AutonomousDrivingVizImpl &&) = delete;
  AutonomousDrivingVizImpl &operator=(AutonomousDrivingVizImpl &&) = delete;

  /// Initialize and create the various point cloud and car objects
  bool Init();

  /// Deinitialize
  bool DeInit();

  /// Render all information
  void Render();

public:
  /// Background rendering thread
  std::thread render_thread_;

  /// Helper mutexes and atomic flags
  std::mutex mtx_map_cloud_;
  std::mutex mtx_current_scan_;
  std::mutex mtx_nav_state_;
  std::mutex mtx_gps_pose_;

  std::atomic<bool> exit_flag_;

  std::atomic<bool> cloud_global_need_update_;
  std::atomic<bool> kf_result_need_update_;
  std::atomic<bool> current_scan_need_update_;
  std::atomic<bool> lidarloc_need_update_;
  std::atomic<bool> pgoloc_need_update_;
  std::atomic<bool> gps_need_update_;

  CloudPtr current_scan_ = nullptr; // Current scan
  SE3 current_pose_;                // Pose corresponding to the current scan

  // Map point cloud
  std::map<Vec2i, CloudPtr, less_vec<2>> cloud_global_map_;

  /// gps
  SE3 gps_pose_;

  /// Filter state
  SE3 pose_;
  Vec3d vel_;
  Vec3d bias_acc_;
  Vec3d bias_gyr_;
  Vec3d grav_;

  SE3 T_imu_lidar_;
  int max_size_of_current_scan_ = 2000; // Number of recent scans to keep

  //////////////////////////////// Rendering-related members
  //////////////////////////////
private:
  /// Create OpenGL buffers
  void AllocateBuffer();
  void ReleaseBuffer();

  void CreateDisplayLayout();

  void DrawAll(); // Draw the localization window

  /// Render point clouds and call the various update functions
  void RenderClouds();
  bool UpdateGps();
  bool UpdateGlobalMap();
  bool UpdateState();
  bool UpdateCurrentScan();

  void RenderLabels();

private:
  /// Window layout parameters
  int win_width_ = 1920;
  int win_height_ = 1080;
  static constexpr float cam_focus_ = 5000;
  static constexpr float cam_z_near_ = 1.0;
  static constexpr float cam_z_far_ = 1e10;
  static constexpr int menu_width_ = 200;
  const std::string win_name_ = "SAD.UI";
  const std::string dis_main_name_ = "main";
  const std::string dis_3d_name_ = "Cam 3D";
  const std::string dis_3d_main_name_ = "Cam 3D Main"; // main
  const std::string dis_plot_name_ = "Plot";
  const std::string dis_imgs_name = "Images";

  bool following_loc_ =
      true; // Whether the camera follows the localization result

  // text
  pangolin::GlText gltext_label_global_;

  // camera
  pangolin::OpenGlRenderState s_cam_main_;

  /// Cloud rendering
  ui::UiCar car_{Vec3f(0.2, 0.2, 0.8)}; // White car
  std::map<Vec2i, std::shared_ptr<ui::UiCloud>, less_vec<2>>
      cloud_map_ui_; // Point cloud map used for rendering
  std::shared_ptr<ui::UiCloud> current_scan_ui_; // current scan
  std::deque<std::shared_ptr<ui::UiCloud>>
      scans_; // Queue of retained recent scans

  /// Intermediate variables used during UI rendering
  SE3 T_map_odom_for_lio_traj_ui_;     // Used to display the LIO trajectory
  SE3 T_map_baselink_for_lio_traj_ui_; // Used to display the LIO trajectory

  //  trajectory
  std::shared_ptr<ui::UiTrajectory> traj_lidarloc_ui_ = nullptr;
  std::shared_ptr<ui::UiTrajectory> traj_gps_ui_ = nullptr;

  // Filter-state-related data logger objects
  pangolin::DataLog log_vel_;          // Velocity in the odom frame
  pangolin::DataLog log_vel_baselink_; // Velocity in the baselink frame
  pangolin::DataLog log_bias_acc_;     //
  pangolin::DataLog log_bias_gyr_;     //

  std::unique_ptr<pangolin::Plotter> plotter_vel_ = nullptr;
  std::unique_ptr<pangolin::Plotter> plotter_vel_baselink_ = nullptr;
  std::unique_ptr<pangolin::Plotter> plotter_bias_acc_ = nullptr;
  std::unique_ptr<pangolin::Plotter> plotter_bias_gyr_ = nullptr;
};

} // namespace sad::ui

#endif
