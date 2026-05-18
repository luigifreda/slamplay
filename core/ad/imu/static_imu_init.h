#ifndef SLAM_IN_AUTO_DRIVING_STATIC_IMU_INIT_H
#define SLAM_IN_AUTO_DRIVING_STATIC_IMU_INIT_H

#include "ad/common/eigen_types.h"
#include "ad/imu/imu.h"
#include "ad/nav/odom.h"

#include <deque>

namespace sad {

/**
 * Initializer for an IMU in a level stationary state.
 * Usage: call `AddIMU` and `AddOdom` to feed data, and use `InitSuccess` to
 * check whether initialization succeeded. After success, use the `Get*` methods
 * to retrieve the estimated internal parameters.
 *
 * The initializer attempts to initialize the system every time `AddIMU` is
 * called. When odometry is available, initialization requires the wheel-speed
 * readings to be close to zero; otherwise, the vehicle is assumed to be
 * stationary at the beginning.
 * The initializer collects IMU readings over a period of time and estimates the
 * initial biases and noise parameters according to Section 3.5.4 of the book,
 * for use by the ESKF or other filters.
 */
class StaticIMUInit {
public:
  struct Options {
    Options() {}
    double init_time_seconds_ = 10.0; // Required stationary time.
    int init_imu_queue_max_size_ =
        2000; // Maximum size of the IMU initialization queue.
    int static_odom_pulse_ =
        5; // Wheel-encoder noise threshold for the stationary state.
    double max_static_gyro_var =
        0.5; // Gyroscope measurement variance in the stationary state.
    double max_static_acce_var =
        0.05; // Accelerometer measurement variance in the stationary state.
    double gravity_norm_ = 9.81; // Gravity magnitude.
    bool use_speed_for_static_checking_ =
        true; // Whether to use odometry to determine if the vehicle is
              // stationary.
  };

  /// Constructor.
  StaticIMUInit(Options options = Options()) : options_(options) {}

  /// Add IMU data.
  bool AddIMU(const IMU &imu);
  /// Add wheel-speed data.
  bool AddOdom(const Odom &odom);

  /// Check whether initialization succeeded.
  bool InitSuccess() const { return init_success_; }

  /// Get covariance, bias, and gravity estimates.
  Vec3d GetCovGyro() const { return cov_gyro_; }
  Vec3d GetCovAcce() const { return cov_acce_; }
  Vec3d GetInitBg() const { return init_bg_; }
  Vec3d GetInitBa() const { return init_ba_; }
  Vec3d GetGravity() const { return gravity_; }

private:
  /// Attempt to initialize the system.
  bool TryInit();

  Options options_;                // Configuration options.
  bool init_success_ = false;      // Whether initialization succeeded.
  Vec3d cov_gyro_ = Vec3d::Zero(); // Gyroscope measurement noise covariance
                                   // estimated during initialization.
  Vec3d cov_acce_ = Vec3d::Zero(); // Accelerometer measurement noise covariance
                                   // estimated during initialization.
  Vec3d init_bg_ = Vec3d::Zero();  // Initial gyroscope bias.
  Vec3d init_ba_ = Vec3d::Zero();  // Initial accelerometer bias.
  Vec3d gravity_ = Vec3d::Zero();  // Gravity vector.
  bool is_static_ = false;         // Whether the vehicle is stationary.
  std::deque<IMU> init_imu_deque_; // Data used for initialization.
  double current_time_ = 0.0;      // Current time.
  double init_start_time_ = 0.0;   // Initial stationary timestamp.
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_STATIC_IMU_INIT_H
