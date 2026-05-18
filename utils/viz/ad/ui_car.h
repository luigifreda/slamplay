
#ifndef SAD_UI_CAR_H
#define SAD_UI_CAR_H

#include <pangolin/gl/glvbo.h>

#include "ad/common/eigen_types.h"

namespace sad::ui {

/// Car shown in the UI
class UiCar {
public:
  UiCar(const Vec3f &color) : color_(color) {}

  /// Set the car pose and reset the points stored in GPU memory
  void SetPose(const SE3 &pose);

  /// Render the car
  void Render();

private:
  Vec3f color_;
  pangolin::GlBuffer vbo_; // buffer data
  pangolin::GlBuffer cbo_; // per-vertex colors

  static std::vector<Vec3f> car_vertices_; // Car vertices
  static std::vector<Vec4f> car_colors_;   // Axis colors
};

} // namespace sad::ui

#endif
