//
// Created by xiang on 2021/7/19.
//

#ifndef MAPPING_DR_PRE_INTEG_H
#define MAPPING_DR_PRE_INTEG_H

#include <deque>
#include <fstream>
#include <memory>

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "ad/imu/imu.h"
#include "ad/nav/gnss.h"
#include "ad/nav/odom.h"

#include "imu_preintegration.h"
namespace sad {

/**
 * GINS optimized with preintegration
 *
 * This chapter still uses a two-frame model rather than full-batch optimization
 * (although full optimization would also be possible), making it somewhat
 * closer to an ESKF. IMU measurements are accumulated in the preintegrator as
 * they arrive. Each time RTK arrives, it triggers an optimization and
 * marginalization step. The marginalization result is written into the prior of
 * the next frame. For odometry, the most recent odometry measurement is used as
 * the velocity observation.
 */
class GinsPreInteg {
public:
  /// GINS configuration
  struct Options {
    Options() {}

    Vec3d gravity_ = Vec3d(0, 0, -9.8); // Gravity direction

    /// IMU-related noise parameters are configured inside preintegration
    IMUPreintegration::Options preinteg_options_;

    // Noise
    double bias_gyro_var_ = 1e-6; // Gyro bias random-walk standard deviation
    double bias_acce_var_ =
        1e-4; // Accelerometer bias random-walk standard deviation
    Mat3d bg_rw_info_ =
        Mat3d::Identity(); // Gyro random-walk information matrix
    Mat3d ba_rw_info_ =
        Mat3d::Identity(); // Accelerometer random-walk information matrix

    double gnss_pos_noise_ = 0.1;                  // GNSS position variance
    double gnss_height_noise_ = 0.1;               // GNSS height variance
    double gnss_ang_noise_ = 1.0 * math::kDEG2RAD; // GNSS angle variance
    Mat6d gnss_info_ = Mat6d::Identity();          // 6D GNSS information matrix

    /// Wheel odometry related
    double odom_var_ = 0.05;
    Mat3d odom_info_ = Mat3d::Identity();
    double odom_span_ = 0.1;       // Odometry measurement interval
    double wheel_radius_ = 0.155;  // Wheel radius
    double circle_pulse_ = 1024.0; // Encoder pulses per revolution

    bool verbose_ = true; // Whether to print debug information
  };

  /// Options can be set in the constructor or updated later
  GinsPreInteg(Options options = Options()) : options_(options) {
    SetOptions(options_);
  }

  /**
   * IMU processing function; the initial bias must be set before calling this
   * @param imu IMU measurement
   */
  void AddImu(const IMU &imu);

  /**
   * GNSS processing function
   * @param gnss
   */
  void AddGnss(const GNSS &gnss);

  /**
   * Wheel odometry processing function
   * @param odom
   */
  void AddOdom(const Odom &odom);

  /// Set the GINS configuration; can be called during construction or after
  /// static initialization
  void SetOptions(Options options);

  /**
   * Get the current state
   * If the IMU has not been integrated, return the last optimized state
   * Otherwise, predict the state using IMU integration
   * @return
   */
  NavStated GetState() const;

private:
  // Optimization
  void Optimize();

  Options options_;
  double current_time_ = 0.0; // Current time

  std::shared_ptr<IMUPreintegration> pre_integ_ = nullptr;
  std::shared_ptr<NavStated> last_frame_ = nullptr; // Previous state
  std::shared_ptr<NavStated> this_frame_ = nullptr; // Current state
  Mat15d prior_info_ = Mat15d::Identity() * 1e2;    // Current prior

  /// GNSS observations for two frames
  GNSS last_gnss_;
  GNSS this_gnss_;

  IMU last_imu_;   // IMU at the previous time step
  Odom last_odom_; // Odometry at the previous time step
  bool last_odom_set_ = false;

  /// Flags
  bool first_gnss_received_ =
      false; // Whether the first GNSS signal has been received
  bool first_imu_received_ =
      false; // Whether the first IMU signal has been received
};
} // namespace sad

#endif // MAPPING_DR_PRE_INTEG_H
