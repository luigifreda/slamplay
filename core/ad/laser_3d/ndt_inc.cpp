#include "ad/laser_3d/ndt_inc.h"
#include "ad/common/math_utils.h"
#include "ad/pointcloud/lidar_utils.h"
#include "ad/timer/timer.h"

#include <execution>
#include <glog/logging.h>
#include <set>

namespace sad {

void IncNdt3d::AddCloud(CloudPtr cloud_world) {
  std::set<KeyType, less_vec<3>>
      active_voxels; // track which voxels are updated
  for (const auto &p : cloud_world->points) {
    auto pt = ToVec3d(p);
    auto key = CastToInt(Vec3d(pt * options_.inv_voxel_size_));
    auto iter = grids_.find(key);
    if (iter == grids_.end()) {
      // voxel does not exist
      data_.push_front({key, {pt}});
      grids_.insert({key, data_.begin()});

      if (data_.size() >= options_.capacity_) {
        // remove the tail entry
        grids_.erase(data_.back().first);
        data_.pop_back();
      }
    } else {
      // voxel exists, add point and update cache
      iter->second->second.AddPoint(pt);
      data_.splice(data_.begin(), data_,
                   iter->second);   // move updated entry to front
      iter->second = data_.begin(); // grids iterator also points to front
    }

    active_voxels.emplace(key);
  }

  // update active_voxels
  std::for_each(std::execution::par_unseq, active_voxels.begin(),
                active_voxels.end(),
                [this](const auto &key) { UpdateVoxel(grids_[key]->second); });
  flag_first_scan_ = false;
}

void IncNdt3d::GenerateNearbyGrids() {
  if (options_.nearby_type_ == NearbyType::CENTER) {
    nearby_grids_.emplace_back(KeyType::Zero());
  } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
    nearby_grids_ = {KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0),
                     KeyType(0, 1, 0), KeyType(0, -1, 0), KeyType(0, 0, -1),
                     KeyType(0, 0, 1)};
  }
}

void IncNdt3d::UpdateVoxel(VoxelData &v) {
  if (flag_first_scan_) {
    if (v.pts_.size() > 1) {
      math::ComputeMeanAndCov(v.pts_, v.mu_, v.sigma_,
                              [this](const Vec3d &p) { return p; });
      v.info_ = (v.sigma_ + Mat3d::Identity() * 1e-3).inverse(); // avoid NaN
    } else {
      v.mu_ = v.pts_[0];
      v.info_ = Mat3d::Identity() * 1e2;
    }

    v.ndt_estimated_ = true;
    v.pts_.clear();
    return;
  }

  if (v.ndt_estimated_ && v.num_pts_ > options_.max_pts_in_voxel_) {
    v.pts_.clear();
    return;
  }

  if (!v.ndt_estimated_ && v.pts_.size() > options_.min_pts_in_voxel_) {
    // newly added voxel
    math::ComputeMeanAndCov(v.pts_, v.mu_, v.sigma_,
                            [this](const Vec3d &p) { return p; });
    v.info_ = (v.sigma_ + Mat3d::Identity() * 1e-3).inverse(); // avoid NaN
    v.ndt_estimated_ = true;
    v.pts_.clear();
  } else if (v.ndt_estimated_ && v.pts_.size() > options_.min_pts_in_voxel_) {
    // already estimated, and new points have arrived
    Vec3d cur_mu, new_mu;
    Mat3d cur_var, new_var;
    math::ComputeMeanAndCov(v.pts_, cur_mu, cur_var,
                            [this](const Vec3d &p) { return p; });
    math::UpdateMeanAndCov(v.num_pts_, v.pts_.size(), v.mu_, v.sigma_, cur_mu,
                           cur_var, new_mu, new_var);

    v.mu_ = new_mu;
    v.sigma_ = new_var;
    v.num_pts_ += v.pts_.size();
    v.pts_.clear();

    // check info
    Eigen::JacobiSVD svd(v.sigma_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vec3d lambda = svd.singularValues();
    if (lambda[1] < lambda[0] * 1e-3) {
      lambda[1] = lambda[0] * 1e-3;
    }

    if (lambda[2] < lambda[0] * 1e-3) {
      lambda[2] = lambda[0] * 1e-3;
    }

    Mat3d inv_lambda =
        Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();
    v.info_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();
  }
}

bool IncNdt3d::AlignNdt(SE3 &init_pose) {
  LOG(INFO) << "aligning with inc ndt, pts: " << source_->size()
            << ", grids: " << grids_.size();
  assert(grids_.empty() == false);

  SE3 pose = init_pose;

  // pre-generate point indices
  int num_residual_per_point = 1;
  if (options_.nearby_type_ == NearbyType::NEARBY6) {
    num_residual_per_point = 7;
  }

  std::vector<int> index(source_->points.size());
  for (int i = 0; i < index.size(); ++i) {
    index[i] = i;
  }

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
            Vec3i real_key = key + nearby_grids_[i];
            auto it = grids_.find(real_key);
            int real_idx = idx * num_residual_per_point + i;
            /// check whether the Gaussian distribution has been estimated
            if (it != grids_.end() && it->second->second.ndt_estimated_) {
              auto &v = it->second->second; // voxel
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
    double total_res = 0;

    int effective_num = 0;

    Mat6d H = Mat6d::Zero();
    Vec6d err = Vec6d::Zero();

    for (int idx = 0; idx < effect_pts.size(); ++idx) {
      if (!effect_pts[idx]) {
        continue;
      }

      total_res += errors[idx].transpose() * infos[idx] * errors[idx];
      effective_num++;

      H += jacobians[idx].transpose() * infos[idx] * jacobians[idx];
      err += -jacobians[idx].transpose() * infos[idx] * errors[idx];
    }

    if (effective_num < options_.min_effective_pts_) {
      LOG(WARNING) << "effective num too small: " << effective_num;
      init_pose = pose;
      return false;
    }

    Vec6d dx = H.inverse() * err;
    pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
    pose.translation() += dx.tail<3>();

    LOG(INFO) << "iter " << iter << " total res: " << total_res
              << ", eff: " << effective_num
              << ", mean res: " << total_res / effective_num
              << ", dxn: " << dx.norm() << ", dx: " << dx.transpose();

    if (dx.norm() < options_.eps_) {
      LOG(INFO) << "converged, dx = " << dx.transpose();
      break;
    }
  }

  init_pose = pose;
  return true;
}

void IncNdt3d::ComputeResidualAndJacobians(const SE3 &input_pose, Mat18d &HTVH,
                                           Vec18d &HTVr) {
  assert(grids_.empty() == false);
  SE3 pose = input_pose;

  // mostly the same as Align above, except z, H, R are returned rather than
  // processed internally
  int num_residual_per_point = 1;
  if (options_.nearby_type_ == NearbyType::NEARBY6) {
    num_residual_per_point = 7;
  }

  std::vector<int> index(source_->points.size());
  for (int i = 0; i < index.size(); ++i) {
    index[i] = i;
  }

  int total_size = index.size() * num_residual_per_point;

  std::vector<bool> effect_pts(total_size, false);
  std::vector<Eigen::Matrix<double, 3, 18>> jacobians(total_size);
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
          Vec3i real_key = key + nearby_grids_[i];
          auto it = grids_.find(real_key);
          int real_idx = idx * num_residual_per_point + i;
          /// check whether the Gaussian distribution has been estimated
          if (it != grids_.end() && it->second->second.ndt_estimated_) {
            auto &v = it->second->second; // voxel
            Vec3d e = qs - v.mu_;

            // check chi2 th
            double res = e.transpose() * v.info_ * e;
            if (std::isnan(res) || res > options_.res_outlier_th_) {
              effect_pts[real_idx] = false;
              continue;
            }

            // build residual
            Eigen::Matrix<double, 3, 18> J;
            J.setZero();
            J.block<3, 3>(0, 0) = Mat3d::Identity(); // w.r.t. p
            J.block<3, 3>(0, 6) =
                -pose.so3().matrix() * SO3::hat(q); // w.r.t. R

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
  double total_res = 0;
  int effective_num = 0;

  HTVH.setZero();
  HTVr.setZero();

  const double info_ratio = 0.01; // info factor contributed by each point

  for (int idx = 0; idx < effect_pts.size(); ++idx) {
    if (!effect_pts[idx]) {
      continue;
    }

    total_res += errors[idx].transpose() * infos[idx] * errors[idx];
    effective_num++;

    HTVH +=
        jacobians[idx].transpose() * infos[idx] * jacobians[idx] * info_ratio;
    HTVr += -jacobians[idx].transpose() * infos[idx] * errors[idx] * info_ratio;
  }

  LOG(INFO) << "effective: " << effective_num;
}

void IncNdt3d::BuildNDTEdges(sad::VertexPose *v,
                             std::vector<EdgeNDT *> &edges) {
  assert(grids_.empty() == false);
  SE3 pose = v->estimate();

  /// overall process is the same as NDT, except the query function is placed
  /// inside the edge, creating edges bound to v
  for (const auto &pt : source_->points) {
    Vec3d q = ToVec3d(pt);
    auto edge = new EdgeNDT(
        v, q, [this](const Vec3d &qs, Vec3d &mu, Mat3d &info) -> bool {
          Vec3i key = CastToInt(Vec3d(qs * options_.inv_voxel_size_));

          auto it = grids_.find(key);
          /// check whether the Gaussian distribution has been estimated
          if (it != grids_.end() && it->second->second.ndt_estimated_) {
            auto &v = it->second->second; // voxel
            mu = v.mu_;
            info = v.info_;
            return true;
          } else {
            return false;
          }
        });

    if (edge->IsValid()) {
      edges.emplace_back(edge);
    } else {
      delete edge;
    }
  }
}

} // namespace sad