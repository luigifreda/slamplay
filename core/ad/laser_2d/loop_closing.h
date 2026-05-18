#ifndef SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H
#define SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H

#include "ad/common/eigen_types.h"
#include "ad/laser_2d/multi_resolution_likelihood_field.h"
#include "ad/laser_2d/submap.h"

#include <fstream>
#include <map>
#include <memory>

namespace sad {

/**
 * Single-threaded loop closing module
 * Performs loop detection based on odometry-estimated poses; if detected,
 * updates submap poses
 *
 * First calls DetectLoopCandidates to detect possible loops in historical maps
 * Then uses MultiResolutionMatching for alignment
 * If alignment succeeds, builds a pose graph for optimization
 * The graph optimization may also reject a loop as incorrect and remove it
 */
class LoopClosing {
public:
  /// A loop constraint
  struct LoopConstraints {
    LoopConstraints(size_t id1, size_t id2, const SE2 &T12)
        : id_submap1_(id1), id_submap2_(id2), T12_(T12) {}
    size_t id_submap1_ = 0;
    size_t id_submap2_ = 0;
    SE2 T12_; // relative pose
    bool valid_ = true;
  };

  LoopClosing();

  /// Add the latest submap, which may still be under construction
  void AddNewSubmap(std::shared_ptr<Submap> submap);

  /// Add a finished submap; must be called after AddNewSubmap
  void AddFinishedSubmap(std::shared_ptr<Submap> submap);

  /// Perform loop detection for a new frame, update its pose and submap poses
  void AddNewFrame(std::shared_ptr<Frame> frame);

  /// Get loop closures between submaps
  std::map<std::pair<size_t, size_t>, LoopConstraints> GetLoops() const {
    return loop_constraints_;
  }

  bool HasNewLoops() const { return has_new_loops_; }

private:
  /// Detect possible loops between the current frame and historical maps
  bool DetectLoopCandidates();

  /// Match the current frame against historical submaps
  void MatchInHistorySubmaps();

  /// Perform pose graph optimization between submaps
  void Optimize();

  std::shared_ptr<Frame> current_frame_ = nullptr;
  size_t last_submap_id_ = 0; // id of the latest submap

  std::map<size_t, std::shared_ptr<Submap>> submaps_; // all submaps

  // mapping from submap to MR field
  std::map<std::shared_ptr<Submap>, std::shared_ptr<MRLikelihoodField>>
      submap_to_field_;

  std::vector<size_t> current_candidates_; // possible loop closure candidates
  std::map<std::pair<size_t, size_t>, LoopConstraints>
      loop_constraints_; // loop constraints, indexed by the two constrained
                         // submaps
  bool has_new_loops_ = false;

  std::ofstream debug_fout_; // debug output

  /// Parameters
  inline static constexpr float candidate_distance_th_ =
      15.0; // distance between candidate frame and submap center
  inline static constexpr int submap_gap_ =
      1; // gap between current scan and nearest submap id
  inline static constexpr double loop_rk_delta_ =
      1.0; // robust kernel threshold for loop detection
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_LOOP_CLOSING_H
