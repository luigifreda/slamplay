//
// Created by xiang on 2022/1/4.
//

#include "utm_utils.h"
#include "ad/common/math_utils.h"

#include <utm_convert/utm.h>

#include <glog/logging.h>

namespace sad {

// convert latitude and longitude to UTM
bool LatLon2UTM(const Vec2d &latlon, UTMCoordinate &utm_coor) {
  long zone = 0;
  char char_north = 0;
  long ret = Convert_Geodetic_To_UTM(
      latlon[0] * math::kDEG2RAD, latlon[1] * math::kDEG2RAD, &zone,
      &char_north, &utm_coor.xy_[0], &utm_coor.xy_[1]);
  utm_coor.zone_ = (int)zone;
  utm_coor.north_ = char_north == 'N';

  return ret == 0;
}

// convert UTM to latitude and longitude
bool UTM2LatLon(const UTMCoordinate &utm_coor, Vec2d &latlon) {
  bool ret = Convert_UTM_To_Geodetic(
      (long)utm_coor.zone_, utm_coor.north_ ? 'N' : 'S', utm_coor.xy_[0],
      utm_coor.xy_[1], &latlon[0], &latlon[1]);
  latlon *= math::kRAD2DEG;
  return ret == 0;
}

// convert GPS to UTM
bool ConvertGps2UTM(GNSS &gps_msg, const Vec2d &antenna_pos,
                    const double &antenna_angle, const Vec3d &map_origin) {
  /// Convert latitude, longitude, and altitude to UTM.
  UTMCoordinate utm_rtk;
  if (!LatLon2UTM(gps_msg.lat_lon_alt_.head<2>(), utm_rtk)) {
    return false;
  }
  utm_rtk.z_ = gps_msg.lat_lon_alt_[2];

  /// Convert the GPS heading to radians.
  double heading = 0;
  if (gps_msg.heading_valid_) {
    heading =
        (90 - gps_msg.heading_) * math::kDEG2RAD; // Convert from NED to ENU.
  }

  /// Transform from TWG to TWB.
  SE3 TBG(SO3::rotZ(antenna_angle * math::kDEG2RAD),
          Vec3d(antenna_pos[0], antenna_pos[1], 0));
  SE3 TGB = TBG.inverse();

  /// If a map origin is provided, subtract it from the coordinates.
  double x = utm_rtk.xy_[0] - map_origin[0];
  double y = utm_rtk.xy_[1] - map_origin[1];
  double z = utm_rtk.z_ - map_origin[2];
  SE3 TWG(SO3::rotZ(heading), Vec3d(x, y, z));
  SE3 TWB = TWG * TGB;

  gps_msg.utm_valid_ = true;
  gps_msg.utm_.xy_[0] = TWB.translation().x();
  gps_msg.utm_.xy_[1] = TWB.translation().y();
  gps_msg.utm_.z_ = TWB.translation().z();

  if (gps_msg.heading_valid_) {
    // Assemble a pose with rotation.
    gps_msg.utm_pose_ = TWB;
  } else {
    // Assemble an SE3 with translation only.
    // Note that when a mounting offset exists, the actual vehicle pose cannot
    // be recovered.
    gps_msg.utm_pose_ = SE3(SO3(), TWB.translation());
  }

  return true;
}

// convert GPS to UTM only translational components
// results is an SE3 with translation only stored in the translation field of
// input gps_msg
bool ConvertGps2UTMOnlyTrans(GNSS &gps_msg) {
  /// Convert latitude, longitude, and altitude to UTM.
  UTMCoordinate utm_rtk;
  LatLon2UTM(gps_msg.lat_lon_alt_.head<2>(), utm_rtk);
  gps_msg.utm_valid_ = true;
  gps_msg.utm_.xy_ = utm_rtk.xy_;
  gps_msg.utm_.z_ = gps_msg.lat_lon_alt_[2];
  gps_msg.utm_pose_ = SE3(
      SO3(), Vec3d(gps_msg.utm_.xy_[0], gps_msg.utm_.xy_[1], gps_msg.utm_.z_));
  return true;
}

} // namespace sad