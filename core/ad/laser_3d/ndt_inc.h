#ifndef SLAM_IN_AUTO_DRIVING_NDT_INC_H
#define SLAM_IN_AUTO_DRIVING_NDT_INC_H

#include "ad/common/eigen_types.h"
#include "ad/g2o/g2o_types.h"
#include "ad/pointcloud/point_types.h"

#include <list>
namespace sad {

/**
 * Incremental version of NDT
 * Internally maintains incremental voxels, automatically removes older voxels,
 * and updates mean and covariance estimates when adding point clouds to voxels
 */
class IncNdt3d {
public:
  enum class NearbyType {
    CENTER,  // center only
    NEARBY6, // up/down/left/right/front/back
  };

  struct Options {
    int max_iteration_ = 4;       // max number of iterations
    double voxel_size_ = 1.0;     // voxel size
    double inv_voxel_size_ = 1.0; // inverse of voxel size
    int min_effective_pts_ = 10;  // min effective points threshold
    int min_pts_in_voxel_ = 5;    // min points per voxel
    int max_pts_in_voxel_ = 50;   // max points per voxel
    double eps_ = 1e-3;           // convergence criterion
    double res_outlier_th_ = 5.0; // outlier rejection threshold
    size_t capacity_ = 100000;    // number of cached voxels

    NearbyType nearby_type_ = NearbyType::NEARBY6;
  };

  using KeyType = Eigen::Matrix<int, 3, 1>; // voxel index

  /// voxel internal structure
  struct VoxelData {
    VoxelData() {}
    VoxelData(const Vec3d &pt) {
      pts_.emplace_back(pt);
      num_pts_ = 1;
    }

    void AddPoint(const Vec3d &pt) {
      pts_.emplace_back(pt);
      if (!ndt_estimated_) {
        num_pts_++;
      }
    }

    std::vector<Vec3d> pts_;      // internal points; mean and covariance are estimated after accumulating enough
    Vec3d mu_ = Vec3d::Zero();    // mean
    Mat3d sigma_ = Mat3d::Zero(); // covariance
    Mat3d info_ = Mat3d::Zero();  // inverse of covariance

    bool ndt_estimated_ = false; // whether NDT has been estimated
    int num_pts_ = 0;            // total number of points, used for updating estimates
  };

  IncNdt3d() {
    options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
    GenerateNearbyGrids();
  }

  IncNdt3d(Options options) : options_(options) {
    options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
    GenerateNearbyGrids();
  }

  /// get some statistics
  int NumGrids() const { return grids_.size(); }

  /// add point cloud to voxels
  void AddCloud(CloudPtr cloud_world);

  /// set the source scan to be aligned
  void SetSource(CloudPtr source) { source_ = source; }

  /// perform NDT alignment using Gauss-Newton method
  bool AlignNdt(SE3 &init_pose);

  /**
   * Compute Jacobian and residual matrices for a given pose, following IEKF notation (8.17, 8.19)
   * @param pose
   * @param HTVH
   * @param HTVr
   */
  void ComputeResidualAndJacobians(const SE3 &pose, Mat18d &HTVH, Vec18d &HTVr);

  /**
   * Build edges based on estimated NDT
   * @param v
   * @param edges
   */
  void BuildNDTEdges(VertexPose *v, std::vector<EdgeNDT *> &edges);

private:
  /// generate nearby grids based on the nearest neighbor type
  void GenerateNearbyGrids();

  /// update voxel internal data; determine estimation based on newly added points and historical estimates
  void UpdateVoxel(VoxelData &v);

  CloudPtr source_ = nullptr;
  Options options_;

  using KeyAndData = std::pair<KeyType, VoxelData>; // predefined type
  std::list<KeyAndData> data_; // actual data, cached and cleaned up
  std::unordered_map<KeyType, std::list<KeyAndData>::iterator, hash_vec<3>>
      grids_;                         // grid data, stores iterators to actual data
  std::vector<KeyType> nearby_grids_; // nearby grids

  bool flag_first_scan_ = true; // first scan requires special handling
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_NDT_INC_H
