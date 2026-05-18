#ifndef IMUTYPES_H
#define IMUTYPES_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

#include "ad/common/eigen_types.h"
#include "ad/imu/imu.h"
#include "ad/nav/nav_state.h"

namespace sad {

/**
 * IMU preintegrator
 *
 * Call Integrate to add new IMU measurements, then use the Get functions
 * to obtain the preintegrated values.
 * The Jacobians can also be obtained from this class and used to build g2o edge
 * types.
 */
class IMUPreintegration {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// Configuration options
  /// The initial bias needs to be set; the rest can remain unchanged
  struct Options {
    Options() {}
    Vec3d init_bg_ = Vec3d::Zero(); // Initial bias
    Vec3d init_ba_ = Vec3d::Zero(); // Initial bias
    double noise_gyro_ = 1e-2;      // Gyro noise standard deviation
    double noise_acce_ = 1e-1;      // Accelerometer noise standard deviation
  };

  IMUPreintegration(Options options = Options());

  /**
   * Insert a new IMU measurement
   * @param imu   IMU measurement
   * @param dt    Time interval
   */
  void Integrate(const IMU &imu, double dt);

  /**
   * Predict the state after integration starting from a given state
   * @param start State at the starting time
   * @return  Predicted state
   */
  NavStated Predict(const NavStated &start,
                    const Vec3d &grav = Vec3d(0, 0, -9.81)) const;

  /// Get the corrected measurements; the bias may differ from the one used
  /// during preintegration, and a first-order correction is applied
  SO3 GetDeltaRotation(const Vec3d &bg);
  Vec3d GetDeltaVelocity(const Vec3d &bg, const Vec3d &ba);
  Vec3d GetDeltaPosition(const Vec3d &bg, const Vec3d &ba);

public:
  double dt_ = 0;                         // Total preintegration time
  Mat9d cov_ = Mat9d::Zero();             // Accumulated noise covariance
  Mat6d noise_gyro_acce_ = Mat6d::Zero(); // Measurement noise covariance

  // Biases
  Vec3d bg_ = Vec3d::Zero();
  Vec3d ba_ = Vec3d::Zero();

  // Preintegrated measurements
  SO3 dR_;
  Vec3d dv_ = Vec3d::Zero();
  Vec3d dp_ = Vec3d::Zero();

  // Jacobian matrices
  Mat3d dR_dbg_ = Mat3d::Zero();
  Mat3d dV_dbg_ = Mat3d::Zero();
  Mat3d dV_dba_ = Mat3d::Zero();
  Mat3d dP_dbg_ = Mat3d::Zero();
  Mat3d dP_dba_ = Mat3d::Zero();
};

} // namespace sad

#endif // IMUTYPES_H
