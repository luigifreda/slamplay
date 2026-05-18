#ifndef SLAM_IN_AUTO_DRIVING_NDT_3D_H
#define SLAM_IN_AUTO_DRIVING_NDT_3D_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

namespace sad {

/**
 * 3D NDT
 */
class Ndt3d {
public:
  enum class NearbyType {
    CENTER,  // center only
    NEARBY6, // up/down/left/right/front/back
  };

  struct Options {
    int max_iteration_ = 20;       // max number of iterations
    double voxel_size_ = 1.0;      // voxel size
    double inv_voxel_size_ = 1.0;  //
    int min_effective_pts_ = 10;   // min effective points threshold
    int min_pts_in_voxel_ = 3;     // min points per voxel
    double eps_ = 1e-2;            // convergence criterion
    double res_outlier_th_ = 20.0; // outlier rejection threshold
    bool remove_centroid_ = false; // whether to compute and remove point cloud centroids?

    NearbyType nearby_type_ = NearbyType::NEARBY6;
  };

  using KeyType = Eigen::Matrix<int, 3, 1>; // voxel index
  struct VoxelData {
    VoxelData() {}
    VoxelData(size_t id) { idx_.emplace_back(id); }

    std::vector<size_t> idx_;     // indices of points in the point cloud
    Vec3d mu_ = Vec3d::Zero();    // mean
    Mat3d sigma_ = Mat3d::Zero(); // covariance
    Mat3d info_ = Mat3d::Zero();  // inverse of covariance
  };

  Ndt3d() {
    options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
    GenerateNearbyGrids();
  }

  Ndt3d(Options options) : options_(options) {
    options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
    GenerateNearbyGrids();
  }

  /// set the target scan
  void SetTarget(CloudPtr target) {
    target_ = target;
    BuildVoxels();

    // compute point cloud centroid
    target_center_ =
        std::accumulate(target->points.begin(), target_->points.end(),
                        Vec3d::Zero().eval(),
                        [](const Vec3d &c, const PointType &pt) -> Vec3d {
                          return c + ToVec3d(pt);
                        }) /
        target_->size();
  }

  /// set the source scan to be aligned
  void SetSource(CloudPtr source) {
    source_ = source;

    source_center_ =
        std::accumulate(source_->points.begin(), source_->points.end(),
                        Vec3d::Zero().eval(),
                        [](const Vec3d &c, const PointType &pt) -> Vec3d {
                          return c + ToVec3d(pt);
                        }) /
        source_->size();
  }

  void SetGtPose(const SE3 &gt_pose) {
    gt_pose_ = gt_pose;
    gt_set_ = true;
  }

  /// perform NDT alignment using Gauss-Newton method
  bool AlignNdt(SE3 &init_pose);

private:
  void BuildVoxels();

  /// generate nearby grids based on the nearest neighbor type
  void GenerateNearbyGrids();

  CloudPtr target_ = nullptr;
  CloudPtr source_ = nullptr;

  Vec3d target_center_ = Vec3d::Zero();
  Vec3d source_center_ = Vec3d::Zero();

  SE3 gt_pose_;
  bool gt_set_ = false;

  Options options_;

  std::unordered_map<KeyType, VoxelData, hash_vec<3>> grids_; // grid data
  std::vector<KeyType> nearby_grids_;                         // nearby grids
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_NDT_3D_H
