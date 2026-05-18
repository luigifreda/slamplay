#ifndef SLAM_IN_AUTO_DRIVING_BFNN_H
#define SLAM_IN_AUTO_DRIVING_BFNN_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include <thread>

namespace sad {

/**
 * Brute-force Nearest Neighbour
 * @param cloud point cloud
 * @param point query point
 * @return index of the nearest point found
 */
int bfnn_point(CloudPtr cloud, const Vec3f &point);

/**
 * Brute-force Nearest Neighbour, k nearest neighbors
 * @param cloud point cloud
 * @param point query point
 * @param k number of neighbors
 * @return indices of the nearest points found
 */
std::vector<int> bfnn_point_k(CloudPtr cloud, const Vec3f &point, int k = 5);

/**
 * Brute-force nearest neighbor for point clouds
 * @param cloud1  target point cloud
 * @param cloud2  query point cloud
 * @param matches correspondences between the two point clouds
 * @return
 */
void bfnn_cloud(CloudPtr cloud1, CloudPtr cloud2,
                std::vector<std::pair<size_t, size_t>> &matches);

/**
 * Brute-force nearest neighbor for point clouds, multi-threaded version
 * @param cloud1
 * @param cloud2
 * @param matches
 */
void bfnn_cloud_mt(CloudPtr cloud1, CloudPtr cloud2,
                   std::vector<std::pair<size_t, size_t>> &matches);

/**
 * Brute-force nearest neighbor for point clouds, multi-threaded version, k
 * nearest neighbors
 * @param cloud1
 * @param cloud2
 * @param matches
 */
void bfnn_cloud_mt_k(CloudPtr cloud1, CloudPtr cloud2,
                     std::vector<std::pair<size_t, size_t>> &matches,
                     int k = 5);
} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_BFNN_H
