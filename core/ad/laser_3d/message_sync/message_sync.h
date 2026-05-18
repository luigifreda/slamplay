#ifndef SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H
#define SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H

#include "ad/imu/imu.h"
#include "ad/pointcloud/cloud_convert.h"
#include "ad/pointcloud/point_types.h"

#include <deque>
#include <glog/logging.h>

namespace sad {

/// IMU data synchronized with LiDAR
struct MeasureGroup {
  MeasureGroup() { this->lidar_.reset(new FullPointCloudType()); };

  double lidar_begin_time_ = 0;  // Start time of the LiDAR packet
  double lidar_end_time_ = 0;    // End time of the LiDAR packet
  FullCloudPtr lidar_ = nullptr; // LiDAR point cloud
  std::deque<IMUPtr> imu_;       // IMU readings from the previous moment to now
};

/**
 * Synchronize laser data with IMU data
 */
class MessageSync {
public:
  using Callback = std::function<void(const MeasureGroup &)>;

  MessageSync(Callback cb) : callback_(cb), conv_(new CloudConvert) {}

  /// Initialize
  void Init(const std::string &yaml);

  /// Process IMU data
  void ProcessIMU(IMUPtr imu) {
    double timestamp = imu->timestamp_;
    if (timestamp < last_timestamp_imu_) {
      LOG(WARNING) << "imu loop back, clear buffer";
      imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(imu);
  }

  /**
   * Process sensor_msgs::PointCloud2 point cloud
   * @param msg
   */
  void ProcessCloud(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
      LOG(ERROR) << "lidar loop back, clear buffer";
      lidar_buffer_.clear();
    }

    FullCloudPtr cloud(new FullPointCloudType());
    conv_->Process(msg, cloud);
    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar_ = msg->header.stamp.toSec();

    Sync();
  }

  /// Process Livox point cloud
  void ProcessCloud(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
      LOG(WARNING) << "lidar loop back, clear buffer";
      lidar_buffer_.clear();
    }

    last_timestamp_lidar_ = msg->header.stamp.toSec();
    FullCloudPtr ptr(new FullPointCloudType());
    conv_->Process(msg, ptr);

    if (ptr->empty()) {
      return;
    }

    lidar_buffer_.emplace_back(ptr);
    time_buffer_.emplace_back(last_timestamp_lidar_);

    Sync();
  }

private:
  /// Try to synchronize IMU and laser data, returns true on success
  bool Sync();

  Callback callback_; // Callback after data synchronization
  std::shared_ptr<CloudConvert> conv_ = nullptr; // Point cloud converter
  std::deque<FullCloudPtr> lidar_buffer_;        // LiDAR data buffer
  std::deque<IMUPtr> imu_buffer_;                // IMU data buffer
  double last_timestamp_imu_ = -1.0;             // Latest IMU timestamp
  double last_timestamp_lidar_ = 0;              // Latest LiDAR timestamp
  std::deque<double> time_buffer_;
  bool lidar_pushed_ = false;
  MeasureGroup measures_;
  double lidar_end_time_ = 0;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H
