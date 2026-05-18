

#ifndef SAD_UI_TRAJECTORY_H
#define SAD_UI_TRAJECTORY_H

#include "ad/common/eigen_types.h"

#include <pangolin/gl/glvbo.h>

namespace sad::ui {

/// Trajectory rendering in the UI
class UiTrajectory {
public:
  UiTrajectory(const Vec3f &color) : color_(color) { pos_.reserve(max_size_); }

  /// Add one trajectory point
  void AddPt(const SE3 &pose);

  /// Render this trajectory
  void Render();

  void Clear() {
    pos_.clear();
    pos_.reserve(max_size_);
    vbo_.Free();
  }

private:
  int max_size_ = 1e6;          // Maximum number of points to keep
  std::vector<Vec3f> pos_;      // Stored trajectory points
  Vec3f color_ = Vec3f::Zero(); // Display color of the trajectory
  pangolin::GlBuffer vbo_;      // Vertex data stored in GPU memory
};

} // namespace sad::ui

#endif // TFUSION_UI_TRAJECTORY_H
