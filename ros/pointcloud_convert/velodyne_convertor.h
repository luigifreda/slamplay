#pragma once

#include "ad/pointcloud/lidar_utils.h"
#include "ad/pointcloud/point_types.h"
#include "packets_parser.h"
#include "velodyne_config.h"

namespace sad::tools {

/// Converts Velodyne output packets into point cloud format
/// Essentially just wraps `PacketsParser`
class VelodyneConvertor {
public:
  explicit VelodyneConvertor(const VelodyneConfig &config = VelodyneConfig());

  /**
   * Convert packet messages into a `FullCloud`
   * At the same time, transform the lidar point cloud into the IMU frame
   * according to the configuration in `velodyne_config_`
   * @param packets_msg
   * @param out_cloud
   */
  void ProcessScan(const PacketsMsgPtr &packets_msg, FullCloudPtr &out_cloud);

private:
  VelodyneConfig velodyne_config_;
  std::shared_ptr<PacketsParser> packets_parser_ = nullptr;
  FullCloudPtr converted_cloud_ = nullptr;
};

} // namespace sad::tools
