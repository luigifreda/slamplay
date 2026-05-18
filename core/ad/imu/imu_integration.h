//
// Created by xiang on 2021/11/5.
//

#ifndef SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H
#define SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H

#include "ad/common/eigen_types.h"
#include "ad/imu/imu.h"
#include "ad/nav/nav_state.h"

namespace sad {

/**
 * This class demonstrates integration using only IMU measurements.
 */
class IMUIntegration {
public:
  IMUIntegration(const Vec3d &gravity, const Vec3d &init_bg,
                 const Vec3d &init_ba)
      : gravity_(gravity), bg_(init_bg), ba_(init_ba) {}

  // Add an IMU reading.
  void AddIMU(const IMU &imu) {
    double dt = imu.timestamp_ - timestamp_;
    if (dt > 0 && dt < 0.1) {
      // Assume the IMU sampling interval is between 0 and 0.1 seconds.
      p_ = p_ + v_ * dt + 0.5 * gravity_ * dt * dt +
           0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt;
      v_ = v_ + R_ * (imu.acce_ - ba_) * dt + gravity_ * dt;
      R_ = R_ * Sophus::SO3d::exp((imu.gyro_ - bg_) * dt);
    }

    // Update the timestamp.
    timestamp_ = imu.timestamp_;
  }

  /// Build a `NavState`.
  NavStated GetNavState() const {
    return NavStated(timestamp_, R_, p_, v_, bg_, ba_);
  }

  SO3 GetR() const { return R_; }
  Vec3d GetV() const { return v_; }
  Vec3d GetP() const { return p_; }

private:
  // Integrated states.
  SO3 R_;
  Vec3d v_ = Vec3d::Zero();
  Vec3d p_ = Vec3d::Zero();

  double timestamp_ = 0.0;

  Vec3d gravity_ = Vec3d(0, 0, -9.8); // Gravity.

  // Biases, provided externally.
  Vec3d bg_ = Vec3d::Zero();
  Vec3d ba_ = Vec3d::Zero();
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_IMU_INTEGRATION_H
