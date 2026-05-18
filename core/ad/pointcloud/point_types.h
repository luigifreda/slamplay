//
// Created by xiang on 2021/8/18.
//

#ifndef SLAM_IN_AUTO_DRIVING_POINT_TYPES_H
#define SLAM_IN_AUTO_DRIVING_POINT_TYPES_H

#include <pcl/impl/pcl_base.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include "ad/common/eigen_types.h"

namespace sad {

// Define point and point-cloud types used in the system
using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using PointVec = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using IndexVec = std::vector<int>;

// Common conversion helpers from point cloud points to Eigen
inline Vec3f ToVec3f(const PointType &pt) { return pt.getVector3fMap(); }
inline Vec3d ToVec3d(const PointType &pt) {
  return pt.getVector3fMap().cast<double>();
}

// Templated type conversion function
template <typename T, int dim>
inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType &pt);

template <>
inline Eigen::Matrix<float, 2, 1> ToEigen<float, 2>(const PointType &pt) {
  return Vec2f(pt.x, pt.y);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3>(const PointType &pt) {
  return Vec3f(pt.x, pt.y, pt.z);
}

template <typename S>
inline PointType ToPointType(const Eigen::Matrix<S, 3, 1> &pt) {
  PointType p;
  p.x = pt.x();
  p.y = pt.y();
  p.z = pt.z();
  return p;
}

/// Full point type containing ring/range and other attributes
struct FullPointType {
  PCL_ADD_POINT4D;
  float range = 0;
  float radius = 0;
  uint8_t intensity = 0;
  uint8_t ring = 0;
  uint8_t angle = 0;
  double time = 0;
  float height = 0;

  inline FullPointType() {}
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Full point cloud type definition
using FullPointCloudType = pcl::PointCloud<FullPointType>;
using FullCloudPtr = FullPointCloudType::Ptr;

inline Vec3f ToVec3f(const FullPointType &pt) { return pt.getVector3fMap(); }
inline Vec3d ToVec3d(const FullPointType &pt) {
  return pt.getVector3fMap().cast<double>();
}

/// Point cloud color type used in UI
using UiPointType = pcl::PointXYZRGBA;
using UiPointCloudType = pcl::PointCloud<UiPointType>;
using UiCloudPtr = UiPointCloudType::Ptr;

} // namespace sad

POINT_CLOUD_REGISTER_POINT_STRUCT(
    sad::FullPointType,
    (float, x, x)(float, y, y)(float, z, z)(float, range, range)(
        float, radius, radius)(std::uint8_t, intensity,
                               intensity)(std::uint16_t, angle, angle)(
        std::uint8_t, ring, ring)(double, time, time)(float, height, height))

namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                      (float, time, time)(std::uint16_t, ring, ring))
// clang-format on

namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
                                  (float, x, x)
                                      (float, y, y)
                                      (float, z, z)
                                      (float, intensity, intensity)
                                      // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                      (std::uint32_t, t, t)
                                      (std::uint16_t, reflectivity, reflectivity)
                                      (std::uint8_t, ring, ring)
                                      (std::uint16_t, ambient, ambient)
                                      (std::uint32_t, range, range)
)
// clang-format on
#endif // SLAM_IN_AUTO_DRIVING_POINT_TYPES_H
