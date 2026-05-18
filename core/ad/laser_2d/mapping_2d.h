#ifndef SLAM_IN_AUTO_DRIVING_MAPPING_2D_H
#define SLAM_IN_AUTO_DRIVING_MAPPING_2D_H

#include "ad/common/eigen_types.h"
#include "ad/laser_2d/frame.h"
#include "ad/pointcloud/lidar_utils.h"

#include <memory>
#include <opencv2/core.hpp>

namespace sad {

class Submap;
class LoopClosing;

/**
 * Main class for 2D lidar mapping
 */
class Mapping2D {
public:
  bool Init(bool with_loop_closing = true);

  /// Single-echo scan
  bool ProcessScan(Scan2d::Ptr scan);

  /// Multi-echo scan
  /// Not used yet
  bool ProcessScan(MultiScan2d::Ptr scan);

  /**
   * Display global map
   * @param max_size maximum width/height of the global map
   * @return global map image
   */
  cv::Mat ShowGlobalMap(int max_size = 500);

private:
  /// Determine whether the current frame is a keyframe
  bool IsKeyFrame();

  /// Add a keyframe
  void AddKeyFrame();

  /// Expand to a new submap
  void ExpandSubmap();

  /// Data members
  size_t frame_id_ = 0;
  size_t keyframe_id_ = 0;
  size_t submap_id_ = 0;

  bool first_scan_ = true;
  std::shared_ptr<Frame> current_frame_ = nullptr;
  std::shared_ptr<Frame> last_frame_ = nullptr;
  SE2 motion_guess_;
  std::shared_ptr<Frame> last_keyframe_ = nullptr;
  std::shared_ptr<Submap> current_submap_ = nullptr;

  std::vector<std::shared_ptr<Submap>> all_submaps_;

  std::shared_ptr<LoopClosing> loop_closing_ = nullptr; // loop closing

  // Parameters
  inline static constexpr double keyframe_pos_th_ =
      0.3; // keyframe position threshold
  inline static constexpr double keyframe_ang_th_ =
      15 * M_PI / 180; // keyframe angle threshold
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_MAPPING_2D_H
