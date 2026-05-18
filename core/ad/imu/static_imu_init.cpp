//
// Created by xiang on 2021/11/11.
//

#include "static_imu_init.h"
#include "ad/common/math_utils.h"

#include <glog/logging.h>

namespace sad {

bool StaticIMUInit::AddIMU(const IMU &imu) {
  if (init_success_) {
    return true;
  }

  if (options_.use_speed_for_static_checking_ && !is_static_) {
    LOG(WARNING) << "Waiting for the vehicle to remain stationary";
    init_imu_deque_.clear();
    return false;
  }

  if (init_imu_deque_.empty()) {
    // Record the initial stationary timestamp.
    init_start_time_ = imu.timestamp_;
  }

  // Append to the initialization queue.
  init_imu_deque_.push_back(imu);

  double init_time =
      imu.timestamp_ - init_start_time_; // Elapsed initialization time.
  if (init_time > options_.init_time_seconds_) {
    // Try the initialization procedure.
    TryInit();
  }

  // Keep the initialization queue bounded.
  while (init_imu_deque_.size() > options_.init_imu_queue_max_size_) {
    init_imu_deque_.pop_front();
  }

  current_time_ = imu.timestamp_;
  return false;
}

bool StaticIMUInit::AddOdom(const Odom &odom) {
  // Check whether the vehicle is stationary.
  if (init_success_) {
    return true;
  }

  if (odom.left_pulse_ < options_.static_odom_pulse_ &&
      odom.right_pulse_ < options_.static_odom_pulse_) {
    is_static_ = true;
  } else {
    is_static_ = false;
  }

  current_time_ = odom.timestamp_;
  return true;
}

bool StaticIMUInit::TryInit() {
  if (init_imu_deque_.size() < 10) {
    return false;
  }

  // Compute the mean and variance.
  Vec3d mean_gyro, mean_acce;
  math::ComputeMeanAndCovDiag(init_imu_deque_, mean_gyro, cov_gyro_,
                              [](const IMU &imu) { return imu.gyro_; });
  math::ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_,
                              [this](const IMU &imu) { return imu.acce_; });

  // Use the mean acceleration direction and set the gravity magnitude to 9.8.
  LOG(INFO) << "mean acce: " << mean_acce.transpose();
  gravity_ = -mean_acce / mean_acce.norm() * options_.gravity_norm_;

  // Recompute the accelerometer covariance.
  math::ComputeMeanAndCovDiag(
      init_imu_deque_, mean_acce, cov_acce_,
      [this](const IMU &imu) { return imu.acce_ + gravity_; });

  // Check the IMU noise level.
  if (cov_gyro_.norm() > options_.max_static_gyro_var) {
    LOG(ERROR) << "Gyroscope measurement noise is too large: "
               << cov_gyro_.norm() << " > "
               << options_.max_static_gyro_var;
    return false;
  }

  if (cov_acce_.norm() > options_.max_static_acce_var) {
    LOG(ERROR) << "Accelerometer measurement noise is too large: "
               << cov_acce_.norm() << " > "
               << options_.max_static_acce_var;
    return false;
  }

  // Estimate the measurement noise and biases.
  init_bg_ = mean_gyro;
  init_ba_ = mean_acce;

  LOG(INFO) << "IMU initialization succeeded, initialization duration = "
            << current_time_ - init_start_time_
            << ", bg = " << init_bg_.transpose()
            << ", ba = " << init_ba_.transpose()
            << ", gyro sq = " << cov_gyro_.transpose()
            << ", acce sq = " << cov_acce_.transpose()
            << ", grav = " << gravity_.transpose()
            << ", norm: " << gravity_.norm();
  LOG(INFO) << "mean gyro: " << mean_gyro.transpose()
            << " acce: " << mean_acce.transpose();
  init_success_ = true;
  return true;
}

} // namespace sad
