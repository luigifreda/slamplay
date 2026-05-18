
#ifndef SLAM_IN_AUTO_DRIVING_PCL_MAP_VIEWER_H
#define SLAM_IN_AUTO_DRIVING_PCL_MAP_VIEWER_H

#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/pointcloud/point_types.h"

namespace sad {

/// Local map viewer based on the PCL viewer
class PCLMapViewer {
public:
  /// Constructor, specifies the voxel size
  PCLMapViewer(const float &leaf_size, bool use_pcl_vis = true)
      : leaf_size_(leaf_size), tmp_cloud_(new PointCloudType),
        local_map_(new PointCloudType) {
    if (use_pcl_vis) {
      viewer_.reset(new pcl::visualization::PCLVisualizer());
      viewer_->addCoordinateSystem(10, "world");
    } else {
      viewer_ = nullptr;
    }
    voxel_filter_.setLeafSize(leaf_size, leaf_size, leaf_size);
  }

  /**
   * Add a pose and its point cloud (world frame)
   * @param pose
   * @param cloud_world
   */
  void SetPoseAndCloud(const SE3 &pose, CloudPtr cloud_world) {
    voxel_filter_.setInputCloud(cloud_world);
    voxel_filter_.filter(*tmp_cloud_);

    *local_map_ += *tmp_cloud_;
    voxel_filter_.setInputCloud(local_map_);
    voxel_filter_.filter(*local_map_);

    if (viewer_ != nullptr) {
      viewer_->removePointCloud("local_map");
      viewer_->removeCoordinateSystem("vehicle");

      pcl::visualization::PointCloudColorHandlerGenericField<PointType>
          fieldColor(local_map_, "z");
      viewer_->addPointCloud<PointType>(local_map_, fieldColor, "local_map");

      Eigen::Affine3f T;
      T.matrix() = pose.matrix().cast<float>();
      viewer_->addCoordinateSystem(5, T, "vehicle");
      viewer_->spinOnce(1);
    }

    if (local_map_->size() > 600000) {
      leaf_size_ *= 1.26;
      voxel_filter_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
      LOG(INFO) << "viewer set leaf size to " << leaf_size_;
    }
  }

  /// Save the map to a PCD file
  void SaveMap(std::string path) {
    if (local_map_->size() > 0) {
      sad::SaveCloudToFile(path, *local_map_);
      LOG(INFO) << "save map to " << path;
    } else {
      LOG(INFO) << "map is empty" << path;
    }
  }

  void Clean() {
    tmp_cloud_->clear();
    local_map_->clear();
  }

  void ClearAndResetLeafSize(const float leaf_size) {
    leaf_size_ = leaf_size;
    local_map_->clear();
    voxel_filter_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  }

  void Spin(int duration_ms = 1) {
    while (!viewer_->wasStopped()) {
      viewer_->spinOnce(duration_ms);
    }
  }

private:
  pcl::VoxelGrid<PointType> voxel_filter_;
  pcl::visualization::PCLVisualizer::Ptr viewer_ = nullptr; // pcl viewer
  float leaf_size_ = 1.0;
  CloudPtr tmp_cloud_; //
  CloudPtr local_map_;
};
} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_PCL_MAP_VIEWER_H
