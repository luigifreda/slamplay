//
// Created by xiang on 2021/11/11.
//

#ifndef SLAM_IN_AUTO_DRIVING_ESKF_HPP
#define SLAM_IN_AUTO_DRIVING_ESKF_HPP

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "ad/imu/imu.h"
#include "ad/nav/gnss.h"
#include "ad/nav/nav_state.h"
#include "ad/nav/odom.h"
#include "ad/pointcloud/point_types.h"


#include <glog/logging.h>
#include <iomanip>

namespace sad {

/**
 * Error-state Kalman filter introduced in Chapter 3 of the book.
 * GNSS observations can be provided, and the GNSS data should be converted to
 * the vehicle frame in advance.
 *
 * This book uses an 18-dimensional ESKF. The scalar type can be specified by
 * `S`, which defaults to `double`. Variable order: `p`, `v`, `R`, `bg`, `ba`,
 * `grav`, matching the book.
 * @tparam S    scalar precision of the state variables, either `float` or
 * `double`
 */
template <typename S = double> class ESKF {
public:
  /// Type aliases.
  using SO3 = Sophus::SO3<S>;                    // Rotation type.
  using VecT = Eigen::Matrix<S, 3, 1>;           // Vector type.
  using Vec18T = Eigen::Matrix<S, 18, 1>;        // 18-dimensional vector type.
  using Mat3T = Eigen::Matrix<S, 3, 3>;          // 3x3 matrix type.
  using MotionNoiseT = Eigen::Matrix<S, 18, 18>; // Motion noise type.
  using OdomNoiseT = Eigen::Matrix<S, 3, 3>;     // Odometry noise type.
  using GnssNoiseT = Eigen::Matrix<S, 6, 6>;     // GNSS noise type.
  using Mat18T = Eigen::Matrix<S, 18, 18>; // 18-dimensional covariance type.
  using NavStateT = NavState<S>;           // Full nominal-state type.

  struct Options {
    Options() = default;

    /// IMU measurement and bias parameters.
    double imu_dt_ = 0.01; // IMU sampling interval.
    // NOTE The IMU noise terms are in discrete time, so there is no need to
    // multiply by dt again. They can be provided by the initializer.
    double gyro_var_ = 1e-5; // Gyroscope measurement standard deviation.
    double acce_var_ = 1e-2; // Accelerometer measurement standard deviation.
    double bias_gyro_var_ =
        1e-6; // Gyroscope bias random-walk standard deviation.
    double bias_acce_var_ =
        1e-4; // Accelerometer bias random-walk standard deviation.

    /// Odometry parameters.
    double odom_var_ = 0.5;
    double odom_span_ = 0.1;       // Odometry measurement interval.
    double wheel_radius_ = 0.155;  // Wheel radius.
    double circle_pulse_ = 1024.0; // Encoder pulses per revolution.

    /// RTK observation parameters.
    double gnss_pos_noise_ = 0.1;                  // GNSS position noise.
    double gnss_height_noise_ = 0.1;               // GNSS height noise.
    double gnss_ang_noise_ = 1.0 * math::kDEG2RAD; // GNSS rotation noise.

    /// Other configuration.
    bool update_bias_gyro_ = true; // Whether to update the gyroscope bias.
    bool update_bias_acce_ = true; // Whether to update the accelerometer bias.
  };

  /**
   * Initialize with zero initial bias.
   */
  ESKF(Options option = Options()) : options_(option) { BuildNoise(option); }

  /**
   * Set the initial conditions.
   * @param options noise configuration
   * @param init_bg initial gyroscope bias
   * @param init_ba initial accelerometer bias
   * @param gravity gravity vector
   */
  void SetInitialConditions(Options options, const VecT &init_bg,
                            const VecT &init_ba,
                            const VecT &gravity = VecT(0, 0, -9.8)) {
    BuildNoise(options);
    options_ = options;
    bg_ = init_bg;
    ba_ = init_ba;
    g_ = gravity;
    cov_ = Mat18T::Identity() * 1e-4;
  }

  /// Propagate with IMU measurements.
  bool Predict(const IMU &imu);

  /// Update with wheel-speed observations.
  bool ObserveWheelSpeed(const Odom &odom);

  /// Update with GPS observations.
  bool ObserveGps(const GNSS &gnss);

  /**
   * Update with an SE3 observation.
   * @param pose  observed pose
   * @param trans_noise translational noise
   * @param ang_noise   angular noise
   * @return
   */
  bool ObserveSE3(const SE3 &pose, double trans_noise = 0.1,
                  double ang_noise = 1.0 * math::kDEG2RAD);

  /// Accessors.
  /// Get the full nominal state.
  NavStateT GetNominalState() const {
    return NavStateT(current_time_, R_, p_, v_, bg_, ba_);
  }

  /// Get the SE3 state.
  SE3 GetNominalSE3() const { return SE3(R_, p_); }

  /// Set the state `X`.
  void SetX(const NavStated &x, const Vec3d &grav) {
    current_time_ = x.timestamp_;
    R_ = x.R_;
    p_ = x.p_;
    v_ = x.v_;
    bg_ = x.bg_;
    ba_ = x.ba_;
    g_ = grav;
  }

  /// Set the covariance.
  void SetCov(const Mat18T &cov) { cov_ = cov; }

  /// Get the gravity vector.
  Vec3d GetGravity() const { return g_; }

private:
  void BuildNoise(const Options &options) {
    double ev = options.acce_var_;
    double et = options.gyro_var_;
    double eg = options.bias_gyro_var_;
    double ea = options.bias_acce_var_;

    double ev2 = ev; // * ev;
    double et2 = et; // * et;
    double eg2 = eg; // * eg;
    double ea2 = ea; // * ea;

    // Set the process noise.
    Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2,
        ea2, ea2, 0, 0, 0;

    // Set the odometry noise.
    double o2 = options_.odom_var_ * options_.odom_var_;
    odom_noise_.diagonal() << o2, o2, o2;

    // Set the GNSS noise.
    double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
    double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
    double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
    gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
  }

  /// Update the nominal state and reset the error state.
  void UpdateAndReset() {
    p_ += dx_.template block<3, 1>(0, 0);
    v_ += dx_.template block<3, 1>(3, 0);
    R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

    if (options_.update_bias_gyro_) {
      bg_ += dx_.template block<3, 1>(9, 0);
    }

    if (options_.update_bias_acce_) {
      ba_ += dx_.template block<3, 1>(12, 0);
    }

    g_ += dx_.template block<3, 1>(15, 0);

    ProjectCov();
    dx_.setZero();
  }

  /// Project the covariance matrix `P`, following Eq. (3.63).
  void ProjectCov() {
    Mat18T J = Mat18T::Identity();
    J.template block<3, 3>(6, 6) =
        Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0));
    cov_ = J * cov_ * J.transpose();
  }

  /// Member variables.
  double current_time_ = 0.0; // Current time.

  /// Nominal state.
  VecT p_ = VecT::Zero();
  VecT v_ = VecT::Zero();
  SO3 R_;
  VecT bg_ = VecT::Zero();
  VecT ba_ = VecT::Zero();
  VecT g_{0, 0, -9.8};

  /// Error state.
  Vec18T dx_ = Vec18T::Zero();

  /// Covariance matrix.
  Mat18T cov_ = Mat18T::Identity();

  /// Noise matrices.
  MotionNoiseT Q_ = MotionNoiseT::Zero();
  OdomNoiseT odom_noise_ = OdomNoiseT::Zero();
  GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

  /// Flags.
  bool first_gnss_ = true; // Whether this is the first GNSS measurement.

  /// Configuration.
  Options options_;
};

using ESKFD = ESKF<double>;
using ESKFF = ESKF<float>;

template <typename S> bool ESKF<S>::Predict(const IMU &imu) {
  assert(imu.timestamp_ >= current_time_);

  double dt = imu.timestamp_ - current_time_;
  if (dt > (5 * options_.imu_dt_) || dt < 0) {
    // The time interval is invalid, possibly because this is the first IMU
    // sample and there is no history yet.
    LOG(INFO) << "skip this imu because dt_ = " << dt;
    current_time_ = imu.timestamp_;
    return false;
  }

  // Propagate the nominal state.
  VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt +
               0.5 * g_ * dt * dt;
  VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt;
  SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt);

  R_ = new_R;
  v_ = new_v;
  p_ = new_p;
  // The remaining state dimensions stay unchanged.

  // Propagate the error state.
  // Compute the motion-model Jacobian F; see Eq. (3.47).
  // F is actually sparse. It could be applied in expanded form without
  // explicitly building the matrix, but the matrix form is kept here for
  // teaching purposes.
  Mat18T F = Mat18T::Identity();                         // Main diagonal.
  F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt; // p with respect to v.
  F.template block<3, 3>(3, 6) =
      -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt; // v with respect to theta.
  F.template block<3, 3>(3, 12) = -R_.matrix() * dt; // v with respect to ba.
  F.template block<3, 3>(3, 15) =
      Mat3T::Identity() * dt; // v with respect to g.
  F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt)
                                     .matrix(); // theta with respect to theta.
  F.template block<3, 3>(6, 9) =
      -Mat3T::Identity() * dt; // theta with respect to bg.

  // Predict the mean and covariance.
  dx_ = F * dx_; // This line is not strictly necessary: `dx_` should be zero
                 // after reset, so it could be skipped. It is kept because `F`
                 // is still needed for the covariance update.
  cov_ = F * cov_.eval() * F.transpose() + Q_;
  current_time_ = imu.timestamp_;
  return true;
}

template <typename S> bool ESKF<S>::ObserveWheelSpeed(const Odom &odom) {
  assert(odom.timestamp_ >= current_time_);
  // Odometry correction and Jacobian.
  // Use a 3D wheel-speed observation; `H` is 3x18 and mostly zeros.
  Eigen::Matrix<S, 3, 18> H = Eigen::Matrix<S, 3, 18>::Zero();
  H.template block<3, 3>(0, 3) = Mat3T::Identity();

  // Kalman gain.
  Eigen::Matrix<S, 18, 3> K =
      cov_ * H.transpose() * (H * cov_ * H.transpose() + odom_noise_).inverse();

  // Velocity observation.
  double velo_l = options_.wheel_radius_ * odom.left_pulse_ /
                  options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
  double velo_r = options_.wheel_radius_ * odom.right_pulse_ /
                  options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
  double average_vel = 0.5 * (velo_l + velo_r);

  VecT vel_odom(average_vel, 0.0, 0.0);
  VecT vel_world = R_ * vel_odom;

  dx_ = K * (vel_world - v_);

  // Update the covariance.
  cov_ = (Mat18T::Identity() - K * H) * cov_;

  UpdateAndReset();
  return true;
}

template <typename S> bool ESKF<S>::ObserveGps(const GNSS &gnss) {
  /// GNSS observation update.
  assert(gnss.unix_time_ >= current_time_);

  if (first_gnss_) {
    R_ = gnss.utm_pose_.so3();
    p_ = gnss.utm_pose_.translation();
    first_gnss_ = false;
    current_time_ = gnss.unix_time_;
    return true;
  }

  assert(gnss.heading_valid_);
  ObserveSE3(gnss.utm_pose_, options_.gnss_pos_noise_,
             options_.gnss_ang_noise_);
  current_time_ = gnss.unix_time_;

  return true;
}

template <typename S>
bool ESKF<S>::ObserveSE3(const SE3 &pose, double trans_noise,
                         double ang_noise) {
  /// Observation with both rotation and translation.
  /// Observe `p` and `R` in the state vector; `H` is 6x18 and zeros elsewhere.
  Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero();
  H.template block<3, 3>(0, 0) = Mat3T::Identity(); // Position part.
  H.template block<3, 3>(3, 6) =
      Mat3T::Identity(); // Rotation part, Eq. (3.66).

  // Kalman gain and update.
  Vec6d noise_vec;
  noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise,
      ang_noise;

  Mat6d V = noise_vec.asDiagonal();
  Eigen::Matrix<S, 18, 6> K =
      cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse();

  // Update `x` and the covariance.
  Vec6d innov = Vec6d::Zero();
  innov.template head<3>() = (pose.translation() - p_); // Translational part.
  innov.template tail<3>() =
      (R_.inverse() * pose.so3()).log(); // Rotational part, Eq. (3.67).

  dx_ = K * innov;
  cov_ = (Mat18T::Identity() - K * H) * cov_;

  UpdateAndReset();
  return true;
}

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_ESKF_HPP
