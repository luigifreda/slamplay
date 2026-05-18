#ifndef SLAM_IN_AUTO_DRIVING_IESKF_HPP
#define SLAM_IN_AUTO_DRIVING_IESKF_HPP

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "ad/imu/imu.h"
#include "ad/nav/nav_state.h"

namespace sad {

/**
 * Iterated ESKF; motion model matches ch. 3.
 *
 * @tparam S
 */
template <typename S> class IESKF {
public:
  using SO3 = Sophus::SO3<S>;                    // rotation type
  using VecT = Eigen::Matrix<S, 3, 1>;           // 3-vector type
  using Vec18T = Eigen::Matrix<S, 18, 1>;        // 18-vector type
  using Mat3T = Eigen::Matrix<S, 3, 3>;          // 3x3 matrix type
  using MotionNoiseT = Eigen::Matrix<S, 18, 18>; // motion noise type
  using OdomNoiseT = Eigen::Matrix<S, 3, 3>;     // odometry noise type
  using GnssNoiseT = Eigen::Matrix<S, 6, 6>;     // GNSS noise type
  using Mat18T = Eigen::Matrix<S, 18, 18>;       // 18x18 covariance type
  using NavStateT = NavState<S>;                 // nominal navigation state

  struct Options {
    Options() = default;
    /// IEKF settings
    int num_iterations_ = 3; // max iterations
    double quit_eps_ = 1e-3; // stop when ||dx|| below this

    /// IMU measurement and bias random walk
    double imu_dt_ = 0.01;        // nominal IMU interval (s)
    double gyro_var_ = 1e-5;      // gyro measurement std dev
    double acce_var_ = 1e-2;      // accelerometer measurement std dev
    double bias_gyro_var_ = 1e-6; // gyro bias random-walk std dev
    double bias_acce_var_ = 1e-4; // accelerometer bias random-walk std dev

    /// RTK / GNSS observation noise
    double gnss_pos_noise_ = 0.1;                  // GNSS position noise
    double gnss_height_noise_ = 0.1;               // GNSS height noise
    double gnss_ang_noise_ = 1.0 * math::kDEG2RAD; // GNSS orientation noise

    /// other options
    bool update_bias_gyro_ = true; // update gyro bias in state
    bool update_bias_acce_ = true; // update accel bias in state
  };

  /**
   * Default constructor; initial biases are zero.
   */
  IESKF(Options option = Options()) : options_(option) { BuildNoise(option); }

  /**
   * Constructor with externally supplied initial biases.
   * @param init_bg
   * @param init_ba
   * @param gravity
   */
  IESKF(Options options, const VecT &init_bg, const VecT &init_ba,
        const VecT &gravity = VecT(0, 0, -9.8))
      : options_(options) {
    BuildNoise(options);
    bg_ = init_bg;
    ba_ = init_ba;
    g_ = gravity;
  }

  /// set initial conditions
  void SetInitialConditions(Options options, const VecT &init_bg,
                            const VecT &init_ba,
                            const VecT &gravity = VecT(0, 0, -9.8)) {
    BuildNoise(options);
    options_ = options;
    bg_ = init_bg;
    ba_ = init_ba;
    g_ = gravity;

    cov_ = 1e-4 * Mat18T::Identity();
    cov_.template block<3, 3>(6, 6) = 0.1 * math::kDEG2RAD * Mat3T::Identity();
  }

  /// propagate with one IMU sample
  bool Predict(const IMU &imu);

  /**
   * Custom observation (e.g. NDT): given SE3 pose, fill H^T V^{-1} H and H^T V^{-1} r
   * (eq. 8.10 in the book); both can be accumulated in sum form.
   */
  using CustomObsFunc = std::function<void(const SE3 &input_pose,
                                           Eigen::Matrix<S, 18, 18> &HT_Vinv_H,
                                           Eigen::Matrix<S, 18, 1> &HT_Vinv_r)>;

  /// update filter with a custom observation callback
  bool UpdateUsingCustomObserve(CustomObsFunc obs);

  /// accessors
  /// full nominal state
  NavStateT GetNominalState() const {
    return NavStateT(current_time_, R_, p_, v_, bg_, ba_);
  }

  /// pose as SE3
  SE3 GetNominalSE3() const { return SE3(R_, p_); }

  void SetX(const NavStated &x) {
    current_time_ = x.timestamp_;
    R_ = x.R_;
    p_ = x.p_;
    v_ = x.v_;
    bg_ = x.bg_;
    ba_ = x.ba_;
  }

  void SetCov(const Mat18T &cov) { cov_ = cov; }
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

    // set Q
    Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2,
        ea2, ea2, 0, 0, 0;

    double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
    double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
    double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
    gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
  }

  /// apply error state increment to nominal state
  void Update() {
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
  }

  double current_time_ = 0.0;

  // nominal state
  SO3 R_;
  VecT p_ = VecT::Zero();
  VecT v_ = VecT::Zero();
  VecT bg_ = VecT::Zero();
  VecT ba_ = VecT::Zero();
  VecT g_{0, 0, -9.8};

  // error state
  Vec18T dx_ = Vec18T::Zero();

  // covariance
  Mat18T cov_ = Mat18T::Identity();

  // noise
  MotionNoiseT Q_ = MotionNoiseT::Zero();
  GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

  Options options_;
};

using IESKFD = IESKF<double>;
using IESKFF = IESKF<float>;

template <typename S> bool IESKF<S>::Predict(const IMU &imu) {
  /// predict step (same as ESKF in ch. 3)
  assert(imu.timestamp_ >= current_time_);

  double dt = imu.timestamp_ - current_time_;
  if (dt > (5 * options_.imu_dt_) || dt < 0) {
    LOG(INFO) << "skip this imu because dt_ = " << dt;
    current_time_ = imu.timestamp_;
    return false;
  }

  VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt +
               0.5 * g_ * dt * dt;
  VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt;
  SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt);

  R_ = new_R;
  v_ = new_v;
  p_ = new_p;

  Mat18T F = Mat18T::Identity();
  F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;
  F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;
  F.template block<3, 3>(3, 12) = -R_.matrix() * dt;
  F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;
  F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();
  F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;

  cov_ = F * cov_ * F.transpose() + Q_;
  current_time_ = imu.timestamp_;
  return true;
}
template <typename S>
bool IESKF<S>::UpdateUsingCustomObserve(IESKF::CustomObsFunc obs) {
  // observation Jacobian / information supplied by user callback

  SO3 start_R = R_;
  Eigen::Matrix<S, 18, 1> HTVr;
  Eigen::Matrix<S, 18, 18> HTVH;
  Eigen::Matrix<S, 18, Eigen::Dynamic> K;
  Mat18T Pk, Qk;

  for (int iter = 0; iter < options_.num_iterations_; ++iter) {
    // evaluate observation at current nominal pose
    obs(GetNominalSE3(), HTVH, HTVr);

    // project covariance for SO3
    Mat18T J = Mat18T::Identity();
    J.template block<3, 3>(6, 6) =
        Mat3T::Identity() - 0.5 * SO3::hat((R_.inverse() * start_R).log());
    Pk = J * cov_ * J.transpose();

    // iterated Kalman update
    Qk = (Pk.inverse() + HTVH).inverse(); // intermediate; used in covariance update
    dx_ = Qk * HTVr;
    // LOG(INFO) << "iter " << iter << " dx = " << dx_.transpose() << ", dxn: "
    // << dx_.norm();

    // merge increment into nominal state
    Update();

    if (dx_.norm() < options_.quit_eps_) {
      break;
    }
  }

  // update P
  cov_ = (Mat18T::Identity() - Qk * HTVH) * Pk;

  // project P
  Mat18T J = Mat18T::Identity();
  Vec3d dtheta = (R_.inverse() * start_R).log();
  J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dtheta);
  cov_ = J * cov_ * J.inverse();

  dx_.setZero();
  return true;
}

} // namespace sad
#endif // SLAM_IN_AUTO_DRIVING_IEKF_HPP
