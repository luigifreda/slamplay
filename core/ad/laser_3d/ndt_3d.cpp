#include "ad/laser_3d/ndt_3d.h"
#include "ad/common/math_utils.h"
#include "ad/pointcloud/lidar_utils.h"

#include <Eigen/SVD>
#include <execution>
#include <glog/logging.h>

namespace sad {

void Ndt3d::BuildVoxels() {
  assert(target_ != nullptr);
  assert(target_->empty() == false);
  grids_.clear();

  /// assign voxels
  std::vector<size_t> index(target_->size());
  std::for_each(index.begin(), index.end(),
                [idx = 0](size_t &i) mutable { i = idx++; });

  std::for_each(index.begin(), index.end(), [this](const size_t &idx) {
    Vec3d pt = ToVec3d(target_->points[idx]) * options_.inv_voxel_size_;
    auto key = CastToInt(pt);
    if (grids_.find(key) == grids_.end()) {
      grids_.insert({key, {idx}});
    } else {
      grids_[key].idx_.emplace_back(idx);
    }
  });

  /// compute mean and covariance for each voxel
  std::for_each(
      std::execution::par_unseq, grids_.begin(), grids_.end(), [this](auto &v) {
        if (v.second.idx_.size() > options_.min_pts_in_voxel_) {
          // require at least 3 points
          math::ComputeMeanAndCov(v.second.idx_, v.second.mu_, v.second.sigma_,
                                  [this](const size_t &idx) {
                                    return ToVec3d(target_->points[idx]);
                                  });
          // SVD: check max and min singular values, clamp min singular value

          Eigen::JacobiSVD svd(v.second.sigma_,
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
          Vec3d lambda = svd.singularValues();
          if (lambda[1] < lambda[0] * 1e-3) {
            lambda[1] = lambda[0] * 1e-3;
          }

          if (lambda[2] < lambda[0] * 1e-3) {
            lambda[2] = lambda[0] * 1e-3;
          }

          Mat3d inv_lambda =
              Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2])
                  .asDiagonal();

          // v.second.info_ = (v.second.sigma_ + Mat3d::Identity() *
          // 1e-3).inverse();  // avoid NaN
          v.second.info_ =
              svd.matrixV() * inv_lambda * svd.matrixU().transpose();
        }
      });

  /// remove voxels with insufficient points
  for (auto iter = grids_.begin(); iter != grids_.end();) {
    if (iter->second.idx_.size() > options_.min_pts_in_voxel_) {
      iter++;
    } else {
      iter = grids_.erase(iter);
    }
  }
}

bool Ndt3d::AlignNdt(SE3 &init_pose) {
  LOG(INFO) << "aligning with ndt";
  assert(grids_.empty() == false);

  SE3 pose = init_pose;
  if (options_.remove_centroid_) {
    pose.translation() =
        target_center_ - source_center_; // set initial translation
    LOG(INFO) << "init trans set to " << pose.translation().transpose();
  }

  // pre-generate point indices
  int num_residual_per_point = 1;
  if (options_.nearby_type_ == NearbyType::NEARBY6) {
    num_residual_per_point = 7;
  }

  std::vector<int> index(source_->points.size());
  for (int i = 0; i < index.size(); ++i) {
    index[i] = i;
  }

  // concurrent code
  int total_size = index.size() * num_residual_per_point;

  for (int iter = 0; iter < options_.max_iteration_; ++iter) {
    std::vector<bool> effect_pts(total_size, false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(total_size);
    std::vector<Vec3d> errors(total_size);
    std::vector<Mat3d> infos(total_size);

    // Gauss-Newton iteration
    // nearest neighbor search, can be parallelized
    std::for_each(
        std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
          auto q = ToVec3d(source_->points[idx]);
          Vec3d qs = pose * q; // q after transformation

          // compute the voxel containing qs and its neighboring voxels
          Vec3i key = CastToInt(Vec3d(qs * options_.inv_voxel_size_));

          for (int i = 0; i < nearby_grids_.size(); ++i) {
            auto key_off = key + nearby_grids_[i];
            auto it = grids_.find(key_off);
            int real_idx = idx * num_residual_per_point + i;
            if (it != grids_.end()) {
              auto &v = it->second; // voxel
              Vec3d e = qs - v.mu_;

              // check chi2 th
              double res = e.transpose() * v.info_ * e;
              if (std::isnan(res) || res > options_.res_outlier_th_) {
                effect_pts[real_idx] = false;
                continue;
              }

              // build residual
              Eigen::Matrix<double, 3, 6> J;
              J.block<3, 3>(0, 0) = -pose.so3().matrix() * SO3::hat(q);
              J.block<3, 3>(0, 3) = Mat3d::Identity();

              jacobians[real_idx] = J;
              errors[real_idx] = e;
              infos[real_idx] = v.info_;
              effect_pts[real_idx] = true;
            } else {
              effect_pts[real_idx] = false;
            }
          }
        });

    // accumulate Hessian and error, compute dx
    // could use reduce for parallelism in principle, but it's cumbersome; using
    // accumulate instead
    double total_res = 0;
    int effective_num = 0;

    Mat6d H = Mat6d::Zero();
    Vec6d err = Vec6d::Zero();

    for (int idx = 0; idx < effect_pts.size(); ++idx) {
      if (!effect_pts[idx]) {
        continue;
      }

      total_res += errors[idx].transpose() * infos[idx] * errors[idx];
      // chi2.emplace_back(errors[idx].transpose() * infos[idx] * errors[idx]);
      effective_num++;

      H += jacobians[idx].transpose() * infos[idx] * jacobians[idx];
      err += -jacobians[idx].transpose() * infos[idx] * errors[idx];
    }

    if (effective_num < options_.min_effective_pts_) {
      LOG(WARNING) << "effective num too small: " << effective_num;
      return false;
    }

    Vec6d dx = H.inverse() * err;
    pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
    pose.translation() += dx.tail<3>();

    LOG(INFO) << "iter " << iter << " total res: " << total_res
              << ", eff: " << effective_num
              << ", mean res: " << total_res / effective_num
              << ", dxn: " << dx.norm() << ", dx: " << dx.transpose();

    if (gt_set_) {
      double pose_error = (gt_pose_.inverse() * pose).log().norm();
      LOG(INFO) << "iter " << iter << " pose error: " << pose_error;
    }

    if (dx.norm() < options_.eps_) {
      LOG(INFO) << "converged, dx = " << dx.transpose();
      break;
    }
  }

  init_pose = pose;
  return true;
}

void Ndt3d::GenerateNearbyGrids() {
  if (options_.nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(KeyType::Zero());
  } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
    nearby_grids_ = {KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0),
                     KeyType(0, 1, 0), KeyType(0, -1, 0), KeyType(0, 0, -1),
                     KeyType(0, 0, 1)};
  }
}

} // namespace sad