
#include "ui_car.h"

namespace sad::ui {

namespace {

constexpr float kAxisLength = 2.0f;
constexpr float kAxisArrowSize = 0.25f;
constexpr float kAxisArrowHalfWidth = 0.5f * kAxisArrowSize;

} // namespace

std::vector<Vec3f> UiCar::car_vertices_ = {
    // clang-format off
     { 0, 0, 0}, { kAxisLength, 0, 0},
     { kAxisLength, 0, 0}, { kAxisLength - kAxisArrowSize,  kAxisArrowHalfWidth, 0},
     { kAxisLength, 0, 0}, { kAxisLength - kAxisArrowSize, -kAxisArrowHalfWidth, 0},

     { 0, 0, 0}, { 0, kAxisLength, 0},
     { 0, kAxisLength, 0}, {  kAxisArrowHalfWidth, kAxisLength - kAxisArrowSize, 0},
     { 0, kAxisLength, 0}, { -kAxisArrowHalfWidth, kAxisLength - kAxisArrowSize, 0},

     { 0, 0, 0}, { 0, 0, kAxisLength},
     { 0, 0, kAxisLength}, {  kAxisArrowHalfWidth, 0, kAxisLength - kAxisArrowSize},
     { 0, 0, kAxisLength}, { -kAxisArrowHalfWidth, 0, kAxisLength - kAxisArrowSize},
    // clang-format on
};

std::vector<Vec4f> UiCar::car_colors_ = {
    // clang-format off
     {1.f, 0.f, 0.f, 1.f}, {1.f, 0.f, 0.f, 1.f},
     {1.f, 0.f, 0.f, 1.f}, {1.f, 0.f, 0.f, 1.f},
     {1.f, 0.f, 0.f, 1.f}, {1.f, 0.f, 0.f, 1.f},

     {0.f, 1.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 1.f},
     {0.f, 1.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 1.f},
     {0.f, 1.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 1.f},

     {0.f, 0.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 1.f},
     {0.f, 0.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 1.f},
     {0.f, 0.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 1.f},
    // clang-format on
};

void UiCar::SetPose(const SE3 &pose) {
  std::vector<Vec3f> pts;
  for (auto &p : car_vertices_) {
    pts.emplace_back(p);
  }

  // Transform into the world frame
  auto pose_f = pose.cast<float>();
  for (auto &pt : pts) {
    pt = pose_f * pt;
  }

  /// Upload to GPU memory
  vbo_ = pangolin::GlBuffer(pangolin::GlArrayBuffer, pts);
  cbo_ = pangolin::GlBuffer(pangolin::GlArrayBuffer, car_colors_);
}

void UiCar::Render() {
  if (vbo_.IsValid()) {
    glLineWidth(2.0f);
    if (cbo_.IsValid()) {
      pangolin::RenderVboCbo(vbo_, cbo_, true, GL_LINES);
    } else {
      glColor3f(color_[0], color_[1], color_[2]);
      pangolin::RenderVbo(vbo_, GL_LINES);
    }
    glLineWidth(1.0f);
  }
}

} // namespace sad::ui