#include "ad/laser_3d/gen_simu_data.h"
#include "ad/common/eigen_types.h"

#include <opencv2/core/core.hpp>

namespace sad {

void GenSimuData::Gen() {
  GenTarget();

  // generate pose and source
  cv::RNG rng;
  gt_pose_ = SE3(SO3::exp(Vec3d(rng.gaussian(options_.pose_rot_sigma_),
                                rng.gaussian(options_.pose_rot_sigma_),
                                rng.gaussian(options_.pose_rot_sigma_))),
                 Vec3d(rng.gaussian(options_.pose_trans_sigma_),
                       rng.gaussian(options_.pose_trans_sigma_),
                       rng.gaussian(options_.pose_trans_sigma_)));

  // generate source
  source_.reset(new PointCloudType);
  for (int i = 0; i < options_.num_points_; ++i) {
    source_->points.emplace_back(
        ToPointType(gt_pose_ * ToVec3d(target_->points[i])));
  }
  source_->width = source_->size();
}

void GenSimuData::GenTarget() {
  cv::RNG rng;

  // generate target
  target_.reset(new PointCloudType());

  for (int i = 0; i < options_.num_points_; ++i) {
    int num_face = rng.uniform(0, 6);
    // randomly generate a point and assign it to one of the 6 faces
    Vec3d pt(rng.uniform(-options_.length_, options_.length_),
             rng.uniform(-options_.width_, options_.width_),
             rng.uniform(-options_.height_, options_.height_));

    if (num_face == 0) {
      pt.z() = -options_.height_;
    } else if (num_face == 1) {
      pt.z() = options_.height_;
    } else if (num_face == 2) {
      pt.x() = -options_.length_;
    } else if (num_face == 3) {
      pt.x() = options_.length_;
    } else if (num_face == 4) {
      pt.y() = -options_.width_;
    } else if (num_face == 5) {
      pt.y() = options_.width_;
    }
    target_->points.emplace_back(ToPointType(pt));
  }

  target_->width = options_.num_points_;
}

} // namespace sad