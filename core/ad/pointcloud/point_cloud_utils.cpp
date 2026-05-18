//
// Created by BowenBZ on 2023/5/10.
//

#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/pointcloud/point_types.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

/// Utility functions for point clouds

namespace sad {

/// Voxel filtering
void VoxelGrid(CloudPtr cloud, float voxel_size) {
  pcl::VoxelGrid<sad::PointType> voxel;
  voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel.setInputCloud(cloud);

  CloudPtr output(new PointCloudType);
  voxel.filter(*output);
  cloud->swap(*output);
}

/// Remove ground points
void RemoveGround(CloudPtr cloud, float z_min) {
  CloudPtr output(new PointCloudType);
  for (const auto &pt : cloud->points) {
    if (pt.z > z_min) {
      output->points.emplace_back(pt);
    }
  }

  output->height = 1;
  output->is_dense = false;
  output->width = output->points.size();
  cloud->swap(*output);
}

/// Write point cloud file
template <typename CloudType>
void SaveCloudToFile(const std::string &filePath, CloudType &cloud) {
  cloud.height = 1;
  cloud.width = cloud.size();
  pcl::io::savePCDFileASCII(filePath, cloud);
}

template void SaveCloudToFile<PointCloudType>(const std::string &filePath,
                                              PointCloudType &cloud);

template void SaveCloudToFile<FullPointCloudType>(const std::string &filePath,
                                                  FullPointCloudType &cloud);

} // namespace sad