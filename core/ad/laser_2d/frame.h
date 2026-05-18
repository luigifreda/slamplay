#ifndef SLAM_IN_AUTO_DRIVING_FRAME_H
#define SLAM_IN_AUTO_DRIVING_FRAME_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/lidar_utils.h"

namespace sad {

/**
 * A single 2D lidar scan
 */
struct Frame {
  Frame() {}
  Frame(Scan2d::Ptr scan) : scan_(scan) {}

  /// Save the current frame to a text file for offline use
  void Dump(const std::string &filename);

  /// Load frame data from file
  void Load(const std::string &filename);

  size_t id_ = 0;              // scan id
  size_t keyframe_id_ = 0;     // keyframe id
  double timestamp_ = 0;       // timestamp, generally unused
  Scan2d::Ptr scan_ = nullptr; // lidar scan data
  SE2 pose_;                   // pose, world to scan, T_w_c
  SE2 pose_submap_;            // pose, submap to scan, T_s_c
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_FRAME_H
