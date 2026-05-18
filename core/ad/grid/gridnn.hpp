#ifndef SLAM_IN_AUTO_DRIVING_GRID2D_HPP
#define SLAM_IN_AUTO_DRIVING_GRID2D_HPP

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "ad/kdtree/bfnn.h"
#include "ad/pointcloud/point_types.h"


#include <execution>
#include <glog/logging.h>
#include <map>

namespace sad {

/**
 * Grid-based nearest neighbor
 * @tparam dim template parameter, use 2D or 3D grid
 */
template <int dim> class GridNN {
public:
  using KeyType = Eigen::Matrix<int, dim, 1>;
  using PtType = Eigen::Matrix<float, dim, 1>;

  enum class NearbyType {
    CENTER, // center only
    // for 2D
    NEARBY4, // up, down, left, right
    NEARBY8, // up, down, left, right + four corners

    // for 3D
    NEARBY6, // up, down, left, right, front, back
  };

  /**
   * Constructor
   * @param resolution grid resolution
   * @param nearby_type neighbor lookup method
   */
  explicit GridNN(float resolution = 0.1,
                  NearbyType nearby_type = NearbyType::NEARBY4)
      : resolution_(resolution), nearby_type_(nearby_type) {
    inv_resolution_ = 1.0 / resolution_;

    // check dim and nearby
    if (dim == 2 && nearby_type_ == NearbyType::NEARBY6) {
      LOG(INFO) << "2D grid does not support nearby6, using nearby4 instead.";
      nearby_type_ = NearbyType::NEARBY4;
    } else if (dim == 3 && (nearby_type_ != NearbyType::NEARBY6 &&
                            nearby_type_ != NearbyType::CENTER)) {
      LOG(INFO) << "3D grid does not support nearby4/8, using nearby6 instead.";
      nearby_type_ = NearbyType::NEARBY6;
    }

    GenerateNearbyGrids();
  }

  /// Set point cloud and build grid
  bool SetPointCloud(CloudPtr cloud);

  /// Get nearest neighbor
  bool GetClosestPoint(const PointType &pt, PointType &closest_pt, size_t &idx);

  /// Match two point clouds
  bool GetClosestPointForCloud(CloudPtr ref, CloudPtr query,
                               std::vector<std::pair<size_t, size_t>> &matches);
  bool
  GetClosestPointForCloudMT(CloudPtr ref, CloudPtr query,
                            std::vector<std::pair<size_t, size_t>> &matches);

private:
  /// Generate nearby grids based on the neighbor type
  void GenerateNearbyGrids();

  /// Convert spatial coordinates to grid
  KeyType Pos2Grid(const PtType &pt);

  float resolution_ = 0.1;      // resolution
  float inv_resolution_ = 10.0; // inverse resolution

  NearbyType nearby_type_ = NearbyType::NEARBY4;
  std::unordered_map<KeyType, std::vector<size_t>, hash_vec<dim>>
      grids_; // grid data
  CloudPtr cloud_;

  std::vector<KeyType> nearby_grids_; // nearby grids
};

// Implementation
template <int dim> bool GridNN<dim>::SetPointCloud(CloudPtr cloud) {
  std::vector<size_t> index(cloud->size());
  std::for_each(index.begin(), index.end(),
                [idx = 0](size_t &i) mutable { i = idx++; });

  std::for_each(index.begin(), index.end(), [&cloud, this](const size_t &idx) {
    auto pt = cloud->points[idx];
    auto key = Pos2Grid(ToEigen<float, dim>(pt));
    if (grids_.find(key) == grids_.end()) {
      grids_.insert({key, {idx}});
    } else {
      grids_[key].emplace_back(idx);
    }
  });

  cloud_ = cloud;
  LOG(INFO) << "grids: " << grids_.size();
  return true;
}

template <int dim>
Eigen::Matrix<int, dim, 1>
GridNN<dim>::Pos2Grid(const Eigen::Matrix<float, dim, 1> &pt) {
  return pt.array().template round().template cast<int>();
  // Eigen::Matrix<int, dim, 1> ret;
  // for (int i = 0; i < dim; ++i) {
  //     ret(i, 0) = round(pt[i] * inv_resolution_);
  // }
  // return ret;
}

template <> void GridNN<2>::GenerateNearbyGrids() {
  if (nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(KeyType::Zero());
  } else if (nearby_type_ == NearbyType::NEARBY4) {
    nearby_grids_ = {Vec2i(0, 0), Vec2i(-1, 0), Vec2i(1, 0), Vec2i(0, 1),
                     Vec2i(0, -1)};
  } else if (nearby_type_ == NearbyType::NEARBY8) {
    nearby_grids_ = {
        Vec2i(0, 0),   Vec2i(-1, 0), Vec2i(1, 0),  Vec2i(0, 1), Vec2i(0, -1),
        Vec2i(-1, -1), Vec2i(-1, 1), Vec2i(1, -1), Vec2i(1, 1),
    };
  }
}

template <> void GridNN<3>::GenerateNearbyGrids() {
  if (nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(KeyType::Zero());
  } else if (nearby_type_ == NearbyType::NEARBY6) {
    nearby_grids_ = {KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0),
                     KeyType(0, 1, 0), KeyType(0, -1, 0), KeyType(0, 0, -1),
                     KeyType(0, 0, 1)};
  }
}

template <int dim>
bool GridNN<dim>::GetClosestPoint(const PointType &pt, PointType &closest_pt,
                                  size_t &idx) {
  // Search for nearest neighbor around pt's grid cell
  std::vector<size_t> idx_to_check;
  auto key = Pos2Grid(ToEigen<float, dim>(pt));

  std::for_each(nearby_grids_.begin(), nearby_grids_.end(),
                [&key, &idx_to_check, this](const KeyType &delta) {
                  auto dkey = key + delta;
                  auto iter = grids_.find(dkey);
                  if (iter != grids_.end()) {
                    idx_to_check.insert(idx_to_check.end(),
                                        iter->second.begin(),
                                        iter->second.end());
                  }
                });

  if (idx_to_check.empty()) {
    return false;
  }

  // brute force nn in cloud_[idx]
  CloudPtr nearby_cloud(new PointCloudType);
  std::vector<size_t> nearby_idx;
  for (auto &idx : idx_to_check) {
    nearby_cloud->points.template emplace_back(cloud_->points[idx]);
    nearby_idx.emplace_back(idx);
  }

  size_t closest_point_idx = bfnn_point(nearby_cloud, ToVec3f(pt));
  idx = nearby_idx.at(closest_point_idx);
  closest_pt = cloud_->points[idx];

  return true;
}

template <int dim>
bool GridNN<dim>::GetClosestPointForCloud(
    CloudPtr ref, CloudPtr query,
    std::vector<std::pair<size_t, size_t>> &matches) {
  matches.clear();
  std::vector<size_t> index(query->size());
  std::for_each(index.begin(), index.end(),
                [idx = 0](size_t &i) mutable { i = idx++; });
  std::for_each(index.begin(), index.end(),
                [this, &matches, &query](const size_t &idx) {
                  PointType cp;
                  size_t cp_idx;
                  if (GetClosestPoint(query->points[idx], cp, cp_idx)) {
                    matches.emplace_back(cp_idx, idx);
                  }
                });

  return true;
}

template <int dim>
bool GridNN<dim>::GetClosestPointForCloudMT(
    CloudPtr ref, CloudPtr query,
    std::vector<std::pair<size_t, size_t>> &matches) {
  matches.clear();
  // Similar to the serial version, but matches must be pre-allocated; failed
  // matches are filled with invalid IDs
  std::vector<size_t> index(query->size());
  std::for_each(index.begin(), index.end(),
                [idx = 0](size_t &i) mutable { i = idx++; });
  matches.resize(index.size());

  std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                [this, &matches, &query](const size_t &idx) {
                  PointType cp;
                  size_t cp_idx;
                  if (GetClosestPoint(query->points[idx], cp, cp_idx)) {
                    matches[idx] = {cp_idx, idx};
                  } else {
                    matches[idx] = {math::kINVALID_ID, math::kINVALID_ID};
                  }
                });

  return true;
}

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_GRID2D_HPP
