#ifndef SLAM_IN_AUTO_DRIVING_IO_UTILS_H
#define SLAM_IN_AUTO_DRIVING_IO_UTILS_H

#include <fstream>
#include <functional>
#include <utility>

#include <glog/logging.h>

#include "ad/imu/imu.h"
#include "ad/io/dataset_type.h"
#include "ad/nav/odom.h"
#include "ad/pointcloud/lidar_utils.h"
#include "ad/pointcloud/point_types.h"

#include "ros/pointcloud_convert/velodyne_convertor.h"
#include <livox_ros_driver/CustomMsg.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/LaserScan.h>

#include "utils/utm/utm_utils.h"

namespace sad {

/**
 * Read the data text file provided with this book and invoke callbacks
 * The text file mainly provides IMU/Odom/GNSS readings
 */
class TxtIO {
public:
  TxtIO(const std::string &file_path) : fin(file_path) {}

  /// Callback type definitions
  using IMUProcessFuncType = std::function<void(const IMU &)>;
  using OdomProcessFuncType = std::function<void(const Odom &)>;
  using GNSSProcessFuncType = std::function<void(const GNSS &)>;

  TxtIO &SetIMUProcessFunc(IMUProcessFuncType imu_proc) {
    imu_proc_ = std::move(imu_proc);
    return *this;
  }

  TxtIO &SetOdomProcessFunc(OdomProcessFuncType odom_proc) {
    odom_proc_ = std::move(odom_proc);
    return *this;
  }

  TxtIO &SetGNSSProcessFunc(GNSSProcessFuncType gnss_proc) {
    gnss_proc_ = std::move(gnss_proc);
    return *this;
  }

  // Iterate through file content and invoke callbacks
  void Go();

private:
  std::ifstream fin;
  IMUProcessFuncType imu_proc_;
  OdomProcessFuncType odom_proc_;
  GNSSProcessFuncType gnss_proc_;
};

/**
 * ROSBAG IO
 * Specify a bag name and register callbacks to iterate messages in order
 */
class RosbagIO {
public:
  explicit RosbagIO(std::string bag_file,
                    DatasetType dataset_type = DatasetType::NCLT)
      : bag_file_(std::move(bag_file)), dataset_type_(dataset_type) {
    assert(dataset_type_ != DatasetType::UNKNOWN);
  }

  using MessageProcessFunction =
      std::function<bool(const rosbag::MessageInstance &m)>;

  /// Some convenient topic/message handler aliases
  using Scan2DHandle = std::function<bool(sensor_msgs::LaserScanPtr)>;
  using MultiScan2DHandle = std::function<bool(MultiScan2d::Ptr)>;
  using PointCloud2Handle = std::function<bool(sensor_msgs::PointCloud2::Ptr)>;
  using FullPointCloudHandle = std::function<bool(FullCloudPtr)>;
  using ImuHandle = std::function<bool(IMUPtr)>;
  using GNSSHandle = std::function<bool(GNSSPtr)>;
  using OdomHandle = std::function<bool(const Odom &)>;
  using LivoxHandle =
      std::function<bool(const livox_ros_driver::CustomMsg::ConstPtr &msg)>;

  // Iterate through bag content and invoke callbacks
  void Go();

  /// Generic handler registration
  RosbagIO &AddHandle(const std::string &topic_name,
                      MessageProcessFunction func) {
    process_func_.emplace(topic_name, func);
    return *this;
  }

  /// 2D lidar handler
  RosbagIO &AddScan2DHandle(const std::string &topic_name, Scan2DHandle f) {
    return AddHandle(topic_name, [f](const rosbag::MessageInstance &m) -> bool {
      auto msg = m.instantiate<sensor_msgs::LaserScan>();
      if (msg == nullptr) {
        return false;
      }
      return f(msg);
    });
  }

  /// Multi-echo 2D lidar handler
  RosbagIO &AddMultiScan2DHandle(const std::string &topic_name,
                                 MultiScan2DHandle f) {
    return AddHandle(topic_name, [f](const rosbag::MessageInstance &m) -> bool {
      auto msg = m.instantiate<MultiScan2d>();
      if (msg == nullptr) {
        return false;
      }
      return f(msg);
    });
  }

  /// Automatically select topic name by dataset type
  RosbagIO &AddAutoPointCloudHandle(PointCloud2Handle f) {
    if (dataset_type_ == DatasetType::WXB_3D) {
      return AddHandle(
          wxb_lidar_topic, [f, this](const rosbag::MessageInstance &m) -> bool {
            auto msg = m.instantiate<PacketsMsg>();
            if (msg == nullptr) {
              return false;
            }

            FullCloudPtr cloud(new FullPointCloudType);
            vlp_parser_.ProcessScan(msg, cloud);
            auto cloud_msg = CloudToPointCloud2Ptr<FullPointType>(cloud);
            return f(cloud_msg);
          });
    } else if (dataset_type_ == DatasetType::AVIA) {
      // AVIA cannot directly provide PointCloud2
      return *this;
    } else {
      return AddHandle(GetLidarTopicName(),
                       [f](const rosbag::MessageInstance &m) -> bool {
                         auto msg = m.instantiate<sensor_msgs::PointCloud2>();
                         if (msg == nullptr) {
                           return false;
                         }
                         return f(msg);
                       });
    }
  }

  /// Automatically process RTK messages by dataset
  RosbagIO &AddAutoRTKHandle(GNSSHandle f) {
    if (dataset_type_ == DatasetType::NCLT) {
      return AddHandle(nclt_rtk_topic,
                       [f](const rosbag::MessageInstance &m) -> bool {
                         auto msg = m.instantiate<sensor_msgs::NavSatFix>();
                         if (msg == nullptr) {
                           return false;
                         }

                         GNSSPtr gnss(new GNSS(msg));
                         ConvertGps2UTMOnlyTrans(*gnss);
                         if (std::isnan(gnss->lat_lon_alt_[2])) {
                           // Seems to contain NaN
                           return false;
                         }

                         return f(gnss);
                       });
    }

    // TODO RTK conversion mapping for other datasets
    return *this;
  }

  /// PointCloud2 handler
  RosbagIO &AddPointCloud2Handle(const std::string &topic_name,
                                 PointCloud2Handle f) {
    return AddHandle(topic_name, [f](const rosbag::MessageInstance &m) -> bool {
      auto msg = m.instantiate<sensor_msgs::PointCloud2>();
      if (msg == nullptr) {
        return false;
      }
      return f(msg);
    });
  }

  /// Livox message handler
  RosbagIO &AddLivoxHandle(LivoxHandle f) {
    return AddHandle(GetLidarTopicName(),
                     [f](const rosbag::MessageInstance &m) -> bool {
                       auto msg = m.instantiate<livox_ros_driver::CustomMsg>();
                       if (msg == nullptr) {
                         LOG(INFO) << "cannot inst: " << m.getTopic();
                         return false;
                       }
                       return f(msg);
                     });
  }

  /// WXB Velodyne packets handler
  RosbagIO &AddVelodyneHandle(const std::string &topic_name,
                              FullPointCloudHandle f) {
    return AddHandle(topic_name,
                     [f, this](const rosbag::MessageInstance &m) -> bool {
                       auto msg = m.instantiate<PacketsMsg>();
                       if (msg == nullptr) {
                         return false;
                       }

                       FullCloudPtr cloud(new FullPointCloudType);
                       vlp_parser_.ProcessScan(msg, cloud);

                       return f(cloud);
                     });
  }

  /// IMU
  RosbagIO &AddImuHandle(ImuHandle f);

  /// Clear existing handlers
  void CleanProcessFunc() { process_func_.clear(); }

private:
  /// Get lidar topic name by configured dataset
  std::string GetLidarTopicName() const;

  /// Get IMU topic name by dataset
  std::string GetIMUTopicName() const;

  std::map<std::string, MessageProcessFunction> process_func_;
  std::string bag_file_;
  DatasetType dataset_type_;

  // packets driver
  tools::VelodyneConvertor vlp_parser_;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_IO_UTILS_H
