//
// Created by xiang on 2022/3/15.
//

#ifndef SLAM_IN_AUTO_DRIVING_LIDAR_UTILS_H
#define SLAM_IN_AUTO_DRIVING_LIDAR_UTILS_H

#define USE_LIDAR_UTILS 1

#if USE_LIDAR_UTILS

#include <cstdint>

#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/MultiEchoLaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>

#include "ad/pointcloud/point_types.h"

#include <velodyne_msgs/VelodyneScan.h>

/// Message definitions and utility functions for lidar scans
using Scan2d = sensor_msgs::LaserScan;
using MultiScan2d = sensor_msgs::MultiEchoLaserScan;
using PacketsMsg = velodyne_msgs::VelodyneScan;
using PacketsMsgPtr = boost::shared_ptr<PacketsMsg>;

namespace sad {

inline pcl::PCLHeader ToPclHeader(const std_msgs::Header &header) {
  pcl::PCLHeader pcl_header;
  pcl_header.seq = header.seq;
  pcl_header.stamp = static_cast<std::uint64_t>(header.stamp.sec) * 1000000ULL +
                     static_cast<std::uint64_t>(header.stamp.nsec) / 1000ULL;
  pcl_header.frame_id = header.frame_id;
  return pcl_header;
}

inline std_msgs::Header ToRosHeader(const pcl::PCLHeader &header) {
  std_msgs::Header ros_header;
  ros_header.seq = header.seq;
  ros_header.stamp.sec = static_cast<uint32_t>(header.stamp / 1000000ULL);
  ros_header.stamp.nsec =
      static_cast<uint32_t>((header.stamp % 1000000ULL) * 1000ULL);
  ros_header.frame_id = header.frame_id;
  return ros_header;
}

inline pcl::PCLPointField
ToPclPointField(const sensor_msgs::PointField &field) {
  pcl::PCLPointField pcl_field;
  pcl_field.name = field.name;
  pcl_field.offset = field.offset;
  pcl_field.datatype = field.datatype;
  pcl_field.count = field.count;
  return pcl_field;
}

inline sensor_msgs::PointField
ToRosPointField(const pcl::PCLPointField &field) {
  sensor_msgs::PointField ros_field;
  ros_field.name = field.name;
  ros_field.offset = field.offset;
  ros_field.datatype = field.datatype;
  ros_field.count = field.count;
  return ros_field;
}

inline pcl::PCLPointCloud2
ToPclPointCloud2(const sensor_msgs::PointCloud2 &msg) {
  pcl::PCLPointCloud2 pcl_msg;
  pcl_msg.header = ToPclHeader(msg.header);
  pcl_msg.height = msg.height;
  pcl_msg.width = msg.width;
  pcl_msg.is_bigendian = msg.is_bigendian;
  pcl_msg.point_step = msg.point_step;
  pcl_msg.row_step = msg.row_step;
  pcl_msg.data.assign(msg.data.begin(), msg.data.end());
  pcl_msg.is_dense = msg.is_dense;
  pcl_msg.fields.reserve(msg.fields.size());
  for (const auto &field : msg.fields) {
    pcl_msg.fields.emplace_back(ToPclPointField(field));
  }
  return pcl_msg;
}

inline sensor_msgs::PointCloud2
ToRosPointCloud2(const pcl::PCLPointCloud2 &msg) {
  sensor_msgs::PointCloud2 ros_msg;
  ros_msg.header = ToRosHeader(msg.header);
  ros_msg.height = msg.height;
  ros_msg.width = msg.width;
  ros_msg.is_bigendian = msg.is_bigendian;
  ros_msg.point_step = msg.point_step;
  ros_msg.row_step = msg.row_step;
  ros_msg.data.assign(msg.data.begin(), msg.data.end());
  ros_msg.is_dense = msg.is_dense;
  ros_msg.fields.reserve(msg.fields.size());
  for (const auto &field : msg.fields) {
    ros_msg.fields.emplace_back(ToRosPointField(field));
  }
  return ros_msg;
}

inline Scan2d::Ptr MultiToScan2d(MultiScan2d::Ptr mscan) {
  Scan2d::Ptr scan(new Scan2d);
  scan->header = mscan->header;
  scan->range_max = mscan->range_max;
  scan->range_min = mscan->range_min;
  scan->angle_increment = mscan->angle_increment;
  scan->angle_max = mscan->angle_max;
  scan->angle_min = mscan->angle_min;
  for (auto r : mscan->ranges) {
    if (r.echoes.empty()) {
      scan->ranges.emplace_back(scan->range_max + 1.0);
    } else {
      scan->ranges.emplace_back(r.echoes[0]);
    }
  }
  for (auto i : mscan->intensities) {
    if (i.echoes.empty()) {
      scan->intensities.emplace_back(0);
    } else {
      scan->intensities.emplace_back(i.echoes[0]);
    }
  }
  scan->scan_time = mscan->scan_time;
  scan->time_increment = mscan->time_increment;

  // limit range max
  scan->range_max = 20.0;
  return scan;
}

/// Convert ROS PointCloud2 to a standard PCL PointCloud
inline CloudPtr PointCloud2ToCloudPtr(sensor_msgs::PointCloud2::Ptr msg) {
  CloudPtr cloud(new PointCloudType);
  const auto pcl_msg = ToPclPointCloud2(*msg);
  pcl::fromPCLPointCloud2(pcl_msg, *cloud);
  return cloud;
}

template <typename PointT>
inline sensor_msgs::PointCloud2::Ptr
CloudToPointCloud2Ptr(typename pcl::PointCloud<PointT>::Ptr cloud) {
  pcl::PCLPointCloud2 pcl_msg;
  pcl::toPCLPointCloud2(*cloud, pcl_msg);
  sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
  *msg = ToRosPointCloud2(pcl_msg);
  return msg;
}

/**
 * Convert other point cloud types to PointType point clouds
 * Most commonly used for converting full point clouds to XYZI point clouds
 * @tparam PointT
 * @param input
 * @return
 */
template <typename PointT = FullPointType>
CloudPtr ConvertToCloud(typename pcl::PointCloud<PointT>::Ptr input) {
  CloudPtr cloud(new PointCloudType);
  for (auto &pt : input->points) {
    PointType p;
    p.x = pt.x;
    p.y = pt.y;
    p.z = pt.z;
    p.intensity = pt.intensity;
    cloud->points.emplace_back(p);
  }
  cloud->width = input->width;
  return cloud;
}

/// Apply voxel filtering to a point cloud with the given resolution
inline CloudPtr VoxelCloud(CloudPtr cloud, float voxel_size = 0.1) {
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel.setInputCloud(cloud);

  CloudPtr output(new PointCloudType);
  voxel.filter(*output);
  return output;
}

template <typename S, int n>
inline Eigen::Matrix<int, n, 1> CastToInt(const Eigen::Matrix<S, n, 1> &value) {
  return value.array().round().template cast<int>();
}

} // namespace sad

#endif // USE_LIDAR_UTILS

#endif // SLAM_IN_AUTO_DRIVING_LIDAR_UTILS_H
