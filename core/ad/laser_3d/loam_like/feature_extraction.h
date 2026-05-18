#ifndef SLAM_IN_AUTO_DRIVING_FEATURE_EXTRACTION_H
#define SLAM_IN_AUTO_DRIVING_FEATURE_EXTRACTION_H

#include "ad/pointcloud/point_types.h"

namespace sad {

/**
 * Feature extraction based on scan line information.
 * Requires knowledge of the lidar's scan line distribution; currently only supports Velodyne lidars.
 */
class FeatureExtraction {
  /// Structure holding a line ID and curvature value
  struct IdAndValue {
    IdAndValue() {}
    IdAndValue(int id, double value) : id_(id), value_(value) {}
    int id_ = 0;
    double value_ = 0; // curvature
  };

public:
  FeatureExtraction() {}

  /**
   * Extract edge points and planar points.
   * @param pc_in         input point cloud (full information)
   * @param pc_out_edge   output point cloud of edge points
   * @param pc_out_surf   output point cloud of planar points
   */
  void Extract(FullCloudPtr pc_in, CloudPtr pc_out_edge, CloudPtr pc_out_surf);

  /**
   * Extract edge and planar points from a single sector.
   * @param pc_in
   * @param cloud_curvature
   * @param pc_out_edge
   * @param pc_out_surf
   */
  void ExtractFromSector(const CloudPtr &pc_in,
                         std::vector<IdAndValue> &cloud_curvature,
                         CloudPtr &pc_out_edge, CloudPtr &pc_out_surf);

private:
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_FEATURE_EXTRACTION_H
