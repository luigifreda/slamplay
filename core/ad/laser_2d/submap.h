#ifndef SLAM_IN_AUTO_DRIVING_SUBMAP_H
#define SLAM_IN_AUTO_DRIVING_SUBMAP_H

#include "ad/laser_2d/frame.h"
#include "ad/laser_2d/likelihood_field.h"
#include "ad/laser_2d/occupancy_map.h"

namespace sad {

/**
 * Submap class
 * A submap is associated with several keyframes and maintains its own occupancy
 * grid map and likelihood field. When adding keyframes to a submap, its
 * occupancy grid and likelihood field are updated. A submap has its own pose
 * (Tws); each frame's world pose = submap pose * frame's relative pose in the
 * submap.
 */
class Submap {
public:
  Submap(const SE2 &pose) : pose_(pose) {
    occu_map_.SetPose(pose_);
    field_.SetPose(pose_);
  }

  /// Copy the occupancy grid from another submap into this one
  void SetOccuFromOtherSubmap(std::shared_ptr<Submap> other);

  /// Match a frame against this submap, compute frame->pose
  bool MatchScan(std::shared_ptr<Frame> frame);

  /// Check whether the current scan has points outside the submap
  bool HasOutsidePoints() const;

  /// Add a frame to the occupancy grid map
  void AddScanInOccupancyMap(std::shared_ptr<Frame> frame);

  /// Add a keyframe to the submap
  void AddKeyFrame(std::shared_ptr<Frame> frame) {
    frames_.emplace_back(frame);
  }

  /// When the submap's pose is updated, reset each frame's world pose
  void UpdateFramePoseWorld();

  /// accessors
  OccupancyMap &GetOccuMap() { return occu_map_; }
  LikelihoodField &GetLikelihood() { return field_; }

  std::vector<std::shared_ptr<Frame>> &GetFrames() { return frames_; }
  size_t NumFrames() const { return frames_.size(); }

  void SetId(size_t id) { id_ = id; }
  size_t GetId() const { return id_; }

  void SetPose(const SE2 &pose);
  SE2 GetPose() const { return pose_; }

private:
  SE2 pose_; // submap pose, Tws
  size_t id_ = 0;

  std::vector<std::shared_ptr<Frame>> frames_; // keyframes in this submap
  LikelihoodField field_;                      // for matching
  OccupancyMap occu_map_; // for generating the occupancy grid map
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_SUBMAP_H
