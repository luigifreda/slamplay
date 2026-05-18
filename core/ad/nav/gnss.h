//
// Created by xiang on 2022/1/4.
//

#ifndef SLAM_IN_AUTO_DRIVING_GNSS_H
#define SLAM_IN_AUTO_DRIVING_GNSS_H

#include "ad/common/eigen_types.h"
#include <sensor_msgs/NavSatFix.h>

namespace sad {

/// GNSS status flag information
/// Usually provided by GNSS vendors; this uses the status flags from Qianxun
enum class GpsStatusType {
  GNSS_FLOAT_SOLUTION = 5,  // Float solution (cm to dm level)
  GNSS_FIXED_SOLUTION = 4,  // Fixed solution (cm level)
  GNSS_PSEUDO_SOLUTION = 2, // Pseudorange differential solution (dm level)
  GNSS_SINGLE_POINT_SOLUTION = 1, // Single-point solution (~10 m level)
  GNSS_NOT_EXIST = 0,             // No GPS signal
  GNSS_OTHER = -1,                // Other
};

/// UTM coordinates
struct UTMCoordinate {
  UTMCoordinate() = default;
  explicit UTMCoordinate(int zone, const Vec2d &xy = Vec2d::Zero(),
                         bool north = true)
      : zone_(zone), xy_(xy), north_(north) {}

  int zone_ = 0;             // UTM zone
  Vec2d xy_ = Vec2d::Zero(); // utm xy
  double z_ = 0;             // z altitude (directly from GPS)
  bool north_ = true;        // Whether in the northern hemisphere
};

/// A GNSS measurement struct
struct GNSS {
  GNSS() = default;
  GNSS(double unix_time, int status, const Vec3d &lat_lon_alt, double heading,
       bool heading_valid)
      : unix_time_(unix_time), lat_lon_alt_(lat_lon_alt), heading_(heading),
        heading_valid_(heading_valid) {
    status_ = GpsStatusType(status);
  }

  /// Convert from ROS NavSatFix
  /// NOTE This only contains position and no heading; convert UTM using code in
  /// ch3
  GNSS(sensor_msgs::NavSatFix::Ptr msg) {
    unix_time_ = msg->header.stamp.toSec();
    // Status flag
    if (int(msg->status.status) >= int(sensor_msgs::NavSatStatus::STATUS_FIX)) {
      status_ = GpsStatusType::GNSS_FIXED_SOLUTION;
    } else {
      status_ = GpsStatusType::GNSS_OTHER;
    }
    // Latitude/longitude/altitude
    lat_lon_alt_ << msg->latitude, msg->longitude, msg->altitude;
  }

  double unix_time_ = 0;                                 // Unix system time
  GpsStatusType status_ = GpsStatusType::GNSS_NOT_EXIST; // GNSS status flag
  Vec3d lat_lon_alt_ =
      Vec3d::Zero(); // Latitude, longitude, altitude; first two are in degrees
  double heading_ = 0.0;       // Heading from dual antennas, in degrees
  bool heading_valid_ = false; // Whether heading is valid

  UTMCoordinate utm_; // UTM coordinates (including zone, etc.)
  bool utm_valid_ =
      false; // Whether UTM is computed (false if lat/lon values are invalid)

  SE3 utm_pose_; // 6DoF pose for post-processing
};

} // namespace sad

using GNSSPtr = std::shared_ptr<sad::GNSS>;

#endif // SLAM_IN_AUTO_DRIVING_GNSS_H
