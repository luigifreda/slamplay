#pragma once

#include <math.h>
#include <iostream>

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

namespace slamplay {

template <typename PointT>
inline typename pcl::PointCloud<PointT>::Ptr createCylinderPointCloud(float radiusx = 0.5, float radiusy = 1.0, float zMin = -1.0, float zMax = 1.0, float zDelta = 0.05) {
    // -----Create example point cloud-----
    typename pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>);

    // We're going to make an ellipse extruded along the z-axis.
    for (float z(zMin); z <= zMax; z += zDelta)
    {
        for (float angle(0.0); angle <= 360.0; angle += 5.0)
        {
            PointT basic_point;
            basic_point.x = radiusx * std::cos(pcl::deg2rad(angle));
            basic_point.y = radiusy * std::sin(pcl::deg2rad(angle));
            basic_point.z = z;
            cloud_ptr->points.push_back(basic_point);
        }
    }

    auto fillTheBase = [&](float baseZ) {
        // Fill the base at zMin
        for (float r(0.0); r <= radiusx; r += zDelta)
        {
            for (float angle(0.0); angle <= 360.0; angle += 5.0)
            {
                PointT basic_point;
                basic_point.x = r * std::cos(pcl::deg2rad(angle));
                basic_point.y = r * radiusy / radiusx * std::sin(pcl::deg2rad(angle));
                basic_point.z = baseZ;
                cloud_ptr->points.push_back(basic_point);
            }
        }
    };
    fillTheBase(zMin);
    fillTheBase(zMax);

    cloud_ptr->width = cloud_ptr->size();
    cloud_ptr->height = 1;
    return cloud_ptr;
}

template <typename PointT>
inline typename pcl::PointCloud<PointT>::Ptr createRandomCubePointCloud(size_t num_points = 100, float box_size = 1.0) {
    typename pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>(num_points, 1));

    // Fill in the CloudIn data
    for (auto& point : *cloud_ptr)
    {
        point.x = box_size * rand() / (RAND_MAX + 1.0f);
        point.y = box_size * rand() / (RAND_MAX + 1.0f);
        point.z = box_size * rand() / (RAND_MAX + 1.0f);
    }
    return cloud_ptr;
}

}  // namespace slamplay