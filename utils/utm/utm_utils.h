//
// Created by xiang on 2022/1/4.
//

#ifndef SLAM_IN_AUTO_DRIVING_UTM_CONVERT_H
#define SLAM_IN_AUTO_DRIVING_UTM_CONVERT_H

#include "ad/nav/gnss.h"
#include "ad/pointcloud/point_types.h"

namespace sad {

/**
 * Compute the UTM pose and 6-DoF pose from the GNSS reading used in this book.
 * @param gnss_reading  input GNSS reading
 * @param antenna_pos   antenna mounting position
 * @param antenna_angle antenna mounting yaw offset
 * @param map_origin    map origin; if provided, it is subtracted from the UTM
 * coordinates
 * @return
 */
bool ConvertGps2UTM(GNSS &gnss_reading, const Vec2d &antenna_pos,
                    const double &antenna_angle,
                    const Vec3d &map_origin = Vec3d::Zero());

/**
 * Convert only the translational latitude/longitude component without applying
 * extrinsics or heading.
 * @param gnss_reading
 * @return
 */
bool ConvertGps2UTMOnlyTrans(GNSS &gnss_reading);

/**
 * Convert latitude/longitude to UTM.
 * NOTE Latitude and longitude are in degrees.
 * @param latlon
 * @param utm_coor
 * @return
 */
bool LatLon2UTM(const Vec2d &latlon, UTMCoordinate &utm_coor);

/**
 * Convert UTM to latitude/longitude.
 * @param utm_coor
 * @param latlon
 * @return
 */
bool UTM2LatLon(const UTMCoordinate &utm_coor, Vec2d &latlon);

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_UTM_CONVERT_H
