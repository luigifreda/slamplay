

#ifndef SAD_UI_CLOUD_H
#define SAD_UI_CLOUD_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include <pangolin/gl/glvbo.h>

namespace sad::ui {

/// Point cloud used in the UI
/// Static point clouds can all be rendered with this class
class UiCloud {
public:
  /// Which color scheme to use for rendering this point cloud
  enum UseColor {
    PCL_COLOR,       // PCL-style color, slightly reddish
    INTENSITY_COLOR, // Intensity
    HEIGHT_COLOR,    // Height
    GRAY_COLOR,      // Render in gray
  };

  UiCloud() {}
  UiCloud(CloudPtr cloud);

  /**
   * Set a UI point cloud from a PCL point cloud
   * @param cloud             PCL point cloud
   * @param pose              Point cloud pose; after setting, it is transformed
   * into the global frame
   */
  void SetCloud(CloudPtr cloud, const SE3 &pose);

  /// Render this point cloud
  void Render();

  void SetRenderColor(UseColor use_color);

private:
  Vec4f IntensityToRgbPCL(const float &intensity) const {
    int index = int(intensity * 6);
    index = index % intensity_color_table_pcl_.size();
    return intensity_color_table_pcl_[index];
  }

  UseColor use_color_ = UseColor::PCL_COLOR;

  std::vector<Vec3f> xyz_data_;             // XYZ buffer
  std::vector<Vec4f> color_data_pcl_;       // color buffer
  std::vector<Vec4f> color_data_intensity_; // color buffer
  std::vector<Vec4f> color_data_height_;    // color buffer
  std::vector<Vec4f> color_data_gray_;      // color buffer

  pangolin::GlBuffer vbo_; // Vertex data stored in GPU memory
  pangolin::GlBuffer cbo_; // Color vertex data

  /// Intensity table used by PCL
  void BuildIntensityTable();
  static std::vector<Vec4f> intensity_color_table_pcl_;
};

} // namespace sad::ui
#endif
