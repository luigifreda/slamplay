//
// Created by xiang on 2021/7/16.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "ad/g2o/g2o_imu_types.h"
#include "ad/g2o/g2o_types.h"
#include "ad/imu/imu_preintegration.h"
#include "ad/imu/static_imu_init.h"
#include "ad/io/io_utils.h"
#include "ad/kf/eskf.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(txt_path, dataDir + "/ad/gnss_nav/10.txt", "Data file path");
DEFINE_double(antenna_angle, 12.06,
              "RTK antenna mounting yaw offset (degrees)");
DEFINE_double(antenna_pox_x, -0.17, "RTK antenna mounting offset X");
DEFINE_double(antenna_pox_y, -0.20, "RTK antenna mounting offset Y");
DEFINE_bool(with_ui, true, "Whether to display the GUI");

/**
 * This program demonstrates how to check the preintegration process.
 * It tests the preintegration under constant angular velocity and constant
 * acceleration.
 * It also tests the preintegration optimization process using ESKF.
 */

TEST(PREINTEGRATION_TEST, ROTATION_TEST) {
  // Test preintegration under constant angular velocity
  double imu_time_span = 0.01; // IMU sampling interval
  Vec3d constant_omega(
      0, 0,
      M_PI); // Angular velocity is 180 deg/s; in 1 s rotation should be 180 deg
  Vec3d gravity(0, 0,
                -9.8); // Z axis points upward, gravity is in negative direction

  sad::NavStated start_status(0), end_status(1.0);
  sad::IMUPreintegration pre_integ;

  // Compare with direct integration
  Sophus::SO3d R;
  Vec3d t = Vec3d::Zero();
  Vec3d v = Vec3d::Zero();

  for (int i = 1; i <= 100; ++i) {
    double time = imu_time_span * i;
    Vec3d acce = -gravity; // The accelerometer should measure an upward force
    pre_integ.Integrate(sad::IMU(time, constant_omega, acce), imu_time_span);

    sad::NavStated this_status = pre_integ.Predict(start_status, gravity);

    t = t + v * imu_time_span + 0.5 * gravity * imu_time_span * imu_time_span +
        0.5 * (R * acce) * imu_time_span * imu_time_span;
    v = v + gravity * imu_time_span + (R * acce) * imu_time_span;
    R = R * Sophus::SO3d::exp(constant_omega * imu_time_span);

    // Verify that direct integration and preintegration are equal in this
    // simple case
    EXPECT_NEAR(t[0], this_status.p_[0], 1e-2);
    EXPECT_NEAR(t[1], this_status.p_[1], 1e-2);
    EXPECT_NEAR(t[2], this_status.p_[2], 1e-2);

    EXPECT_NEAR(v[0], this_status.v_[0], 1e-2);
    EXPECT_NEAR(v[1], this_status.v_[1], 1e-2);
    EXPECT_NEAR(v[2], this_status.v_[2], 1e-2);

    EXPECT_NEAR(R.unit_quaternion().x(), this_status.R_.unit_quaternion().x(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().y(), this_status.R_.unit_quaternion().y(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().z(), this_status.R_.unit_quaternion().z(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().w(), this_status.R_.unit_quaternion().w(),
                1e-4);
  }

  end_status = pre_integ.Predict(start_status);

  LOG(INFO) << "preinteg result: ";
  LOG(INFO) << "end rotation: \n" << end_status.R_.matrix();
  LOG(INFO) << "end trans: \n" << end_status.p_.transpose();
  LOG(INFO) << "end v: \n" << end_status.v_.transpose();

  LOG(INFO) << "direct integ result: ";
  LOG(INFO) << "end rotation: \n" << R.matrix();
  LOG(INFO) << "end trans: \n" << t.transpose();
  LOG(INFO) << "end v: \n" << v.transpose();
  SUCCEED();
}

TEST(PREINTEGRATION_TEST, ACCELERATION_TEST) {
  // Test preintegration under constant acceleration
  double imu_time_span = 0.01; // IMU sampling interval
  Vec3d gravity(0, 0,
                -9.8); // Z axis points upward, gravity is in negative direction
  Vec3d constant_acce(0.1, 0, 0); // Constant acceleration along the x axis

  sad::NavStated start_status(0), end_status(1.0);
  sad::IMUPreintegration pre_integ;

  // Compare with direct integration
  Sophus::SO3d R;
  Vec3d t = Vec3d::Zero();
  Vec3d v = Vec3d::Zero();

  for (int i = 1; i <= 100; ++i) {
    double time = imu_time_span * i;
    Vec3d acce = constant_acce - gravity;
    pre_integ.Integrate(sad::IMU(time, Vec3d::Zero(), acce), imu_time_span);
    sad::NavStated this_status = pre_integ.Predict(start_status, gravity);

    t = t + v * imu_time_span + 0.5 * gravity * imu_time_span * imu_time_span +
        0.5 * (R * acce) * imu_time_span * imu_time_span;
    v = v + gravity * imu_time_span + (R * acce) * imu_time_span;

    // Verify that direct integration and preintegration are equal in this
    // simple case
    EXPECT_NEAR(t[0], this_status.p_[0], 1e-2);
    EXPECT_NEAR(t[1], this_status.p_[1], 1e-2);
    EXPECT_NEAR(t[2], this_status.p_[2], 1e-2);

    EXPECT_NEAR(v[0], this_status.v_[0], 1e-2);
    EXPECT_NEAR(v[1], this_status.v_[1], 1e-2);
    EXPECT_NEAR(v[2], this_status.v_[2], 1e-2);

    EXPECT_NEAR(R.unit_quaternion().x(), this_status.R_.unit_quaternion().x(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().y(), this_status.R_.unit_quaternion().y(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().z(), this_status.R_.unit_quaternion().z(),
                1e-4);
    EXPECT_NEAR(R.unit_quaternion().w(), this_status.R_.unit_quaternion().w(),
                1e-4);
  }

  end_status = pre_integ.Predict(start_status);
  LOG(INFO) << "preinteg result: ";
  LOG(INFO) << "end rotation: \n" << end_status.R_.matrix();
  LOG(INFO) << "end trans: \n" << end_status.p_.transpose();
  LOG(INFO) << "end v: \n" << end_status.v_.transpose();

  LOG(INFO) << "direct integ result: ";
  LOG(INFO) << "end rotation: \n" << R.matrix();
  LOG(INFO) << "end trans: \n" << t.transpose();
  LOG(INFO) << "end v: \n" << v.transpose();
  SUCCEED();
}

void Optimize(sad::NavStated &last_state, sad::NavStated &this_state,
              sad::GNSS &last_gnss, sad::GNSS &this_gnss,
              std::shared_ptr<sad::IMUPreintegration> &preinteg,
              const Vec3d &grav);

/// Use ESKF Predict/Update to verify the preintegration optimization process.
/// In particular, it verifies the preintegration optimization process by
/// comparing the preintegration prediction and the ESKF prediction. The test is
/// performed by comparing the preintegration prediction and the ESKF
/// prediction.
TEST(PREINTEGRATION_TEST, ESKF_TEST) {
  if (fLS::FLAGS_txt_path.empty()) {
    FAIL();
  }

  // Initializer
  sad::StaticIMUInit imu_init; // Use default configuration
  sad::ESKFD eskf;

  sad::TxtIO io(FLAGS_txt_path);
  Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

  std::string resultsPath = resultsDir + "/ad/gnss_nav/gins.txt";
  std::string resultsPathDir =
      std::filesystem::path(resultsPath).parent_path().string();
  if (!std::filesystem::exists(resultsPathDir)) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::ofstream fout(resultsPath);
  bool imu_inited = false, gnss_inited = false;

  /// Set processing callbacks
  bool first_gnss_set = false;
  Vec3d origin = Vec3d::Zero();

  std::shared_ptr<sad::IMUPreintegration> preinteg = nullptr;

  sad::NavStated last_state;
  bool last_state_set = false;

  sad::GNSS last_gnss;
  bool last_gnss_set = false;

  io.SetIMUProcessFunc([&](const sad::IMU &imu) {
      /// IMU processing callback
      if (!imu_init.InitSuccess()) {
        imu_init.AddIMU(imu);
        return;
      }

      /// IMU initialization is required
      if (!imu_inited) {
        // Read initial biases and set up ESKF
        sad::ESKFD::Options options;
        // Noise is estimated by the initializer
        options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
        options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
        eskf.SetInitialConditions(options, imu_init.GetInitBg(),
                                  imu_init.GetInitBa(), imu_init.GetGravity());

        imu_inited = true;
        return;
      }

      if (!gnss_inited) {
        /// Wait for valid RTK data
        return;
      }

      /// Start prediction only after GNSS is also available
      double current_time = eskf.GetNominalState().timestamp_;
      eskf.Predict(imu);

      if (preinteg) {
        preinteg->Integrate(imu, imu.timestamp_ - current_time);

        if (last_state_set) {
          auto pred_of_preinteg =
              preinteg->Predict(last_state, eskf.GetGravity());
          auto pred_of_eskf = eskf.GetNominalState();

          /// The difference between these two predictions should be very small
          EXPECT_NEAR((pred_of_preinteg.p_ - pred_of_eskf.p_).norm(), 0, 1e-2);
          EXPECT_NEAR(
              (pred_of_preinteg.R_.inverse() * pred_of_eskf.R_).log().norm(), 0,
              1e-2);
          EXPECT_NEAR((pred_of_preinteg.v_ - pred_of_eskf.v_).norm(), 0, 1e-2);
        }
      }
    })
      .SetGNSSProcessFunc([&](const sad::GNSS &gnss) {
        /// GNSS processing callback
        if (!imu_inited) {
          return;
        }

        sad::GNSS gnss_convert = gnss;
        if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos,
                                 FLAGS_antenna_angle) ||
            !gnss_convert.heading_valid_) {
          return;
        }

        /// Remove origin offset
        if (!first_gnss_set) {
          origin = gnss_convert.utm_pose_.translation();
          first_gnss_set = true;
        }
        gnss_convert.utm_pose_.translation() -= origin;

        // RTK heading must be valid to be fused into EKF
        auto state_bef_update = eskf.GetNominalState();

        eskf.ObserveGps(gnss_convert);

        // Verify whether the optimization process is correct
        if (last_state_set && last_gnss_set) {
          auto update_state = eskf.GetNominalState();

          LOG(INFO) << "state before eskf update: " << state_bef_update;
          LOG(INFO) << "state after  eskf update: " << update_state;

          auto state_pred = preinteg->Predict(last_state, eskf.GetGravity());
          LOG(INFO) << "state in pred: " << state_pred;

          Optimize(last_state, update_state, last_gnss, gnss_convert, preinteg,
                   eskf.GetGravity());
        }

        last_state = eskf.GetNominalState();
        last_state_set = true;

        // Reset preintegration
        sad::IMUPreintegration::Options options;
        options.init_bg_ = last_state.bg_;
        options.init_ba_ = last_state.ba_;
        preinteg = std::make_shared<sad::IMUPreintegration>(options);

        gnss_inited = true;
        last_gnss = gnss_convert;
        last_gnss_set = true;
      })
      .SetOdomProcessFunc(
          [&](const sad::Odom &odom) { imu_init.AddOdom(odom); })
      .Go();

  SUCCEED();
}

void Optimize(sad::NavStated &last_state, sad::NavStated &this_state,
              sad::GNSS &last_gnss, sad::GNSS &this_gnss,
              std::shared_ptr<sad::IMUPreintegration> &pre_integ,
              const Vec3d &grav) {
  assert(pre_integ != nullptr);

  if (pre_integ->dt_ < 1e-3) {
    // No integration available
    return;
  }

  using BlockSolverType = g2o::BlockSolverX;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

  // Test-only stabilization:
  //
  // Unlike core/ad/imu/gins_pre_integ.cpp (which constrains the previous state
  // with an EdgePriorPoseNavState), in this test,
  // - we can use Gauss-Netwon or Levenberg-Marquardt solver,
  // - we can hard-fixes v0_* vertices to remove gauge freedom and avoid
  // singular Hessians in this minimal graph when using Gauss-Newton solver.
#define USE_GAUSS_NEWTON 1
#define SET_PREV_VERTICES_FIXED 0

#if USE_GAUSS_NEWTON
  auto *solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
#else
  auto *solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
#endif

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  // Previous-time vertices: pose, v, bg, ba
  auto v0_pose = new sad::VertexPose();
  v0_pose->setId(0);
  v0_pose->setEstimate(last_state.GetSE3());
#if SET_PREV_VERTICES_FIXED
  v0_pose->setFixed(true); // fixed the initial pose
#endif
  optimizer.addVertex(v0_pose);

  auto v0_vel = new sad::VertexVelocity();
  v0_vel->setId(1);
  v0_vel->setEstimate(last_state.v_);
#if SET_PREV_VERTICES_FIXED
  v0_vel->setFixed(true); // fixed the initial velocity
#endif
  optimizer.addVertex(v0_vel);

  auto v0_bg = new sad::VertexGyroBias();
  v0_bg->setId(2);
  v0_bg->setEstimate(last_state.bg_);
#if SET_PREV_VERTICES_FIXED
  v0_bg->setFixed(true); // fixed the initial gyro bias
#endif
  optimizer.addVertex(v0_bg);

  auto v0_ba = new sad::VertexAccBias();
  v0_ba->setId(3);
  v0_ba->setEstimate(last_state.ba_);
#if SET_PREV_VERTICES_FIXED
  v0_ba->setFixed(true);
#endif
  optimizer.addVertex(v0_ba);

  // Current-time vertices: pose, v, bg, ba
  auto v1_pose = new sad::VertexPose();
  v1_pose->setId(4);
  v1_pose->setEstimate(this_state.GetSE3());
  optimizer.addVertex(v1_pose);

  auto v1_vel = new sad::VertexVelocity();
  v1_vel->setId(5);
  v1_vel->setEstimate(this_state.v_);
  optimizer.addVertex(v1_vel);

  auto v1_bg = new sad::VertexGyroBias();
  v1_bg->setId(6);
  v1_bg->setEstimate(this_state.bg_);
  optimizer.addVertex(v1_bg);

  auto v1_ba = new sad::VertexAccBias();
  v1_ba->setId(7);
  v1_ba->setEstimate(this_state.ba_);
  optimizer.addVertex(v1_ba);

  // Preintegration edge
  auto edge_inertial = new sad::EdgeInertial(pre_integ, grav);
  edge_inertial->setVertex(0, v0_pose);
  edge_inertial->setVertex(1, v0_vel);
  edge_inertial->setVertex(2, v0_bg);
  edge_inertial->setVertex(3, v0_ba);
  edge_inertial->setVertex(4, v1_pose);
  edge_inertial->setVertex(5, v1_vel);

  auto *rk = new g2o::RobustKernelHuber();
  rk->setDelta(200.0);
  edge_inertial->setRobustKernel(rk);

  optimizer.addEdge(edge_inertial);
  edge_inertial->computeError();
  LOG(INFO) << "inertial init err: " << edge_inertial->chi2();

  auto *edge_gyro_rw = new sad::EdgeGyroRW();
  edge_gyro_rw->setVertex(0, v0_bg);
  edge_gyro_rw->setVertex(1, v1_bg);
  edge_gyro_rw->setInformation(Mat3d::Identity() * 1e6);
  optimizer.addEdge(edge_gyro_rw);

  edge_gyro_rw->computeError();
  LOG(INFO) << "inertial bg rw: " << edge_gyro_rw->chi2();

  auto *edge_acc_rw = new sad::EdgeAccRW();
  edge_acc_rw->setVertex(0, v0_ba);
  edge_acc_rw->setVertex(1, v1_ba);
  edge_acc_rw->setInformation(Mat3d::Identity() * 1e6);
  optimizer.addEdge(edge_acc_rw);

  edge_acc_rw->computeError();
  LOG(INFO) << "inertial ba rw: " << edge_acc_rw->chi2();

#if !SET_PREV_VERTICES_FIXED
  // Prior from the previous time step
  Mat15d prior_info = Mat15d::Identity() * 1e2;
  auto *edge_prior = new sad::EdgePriorPoseNavState(last_state, prior_info);
  edge_prior->setVertex(0, v0_pose);
  edge_prior->setVertex(1, v0_vel);
  edge_prior->setVertex(2, v0_bg);
  edge_prior->setVertex(3, v0_ba);
  optimizer.addEdge(edge_prior);
#endif

  // GNSS edges
  auto edge_gnss0 = new sad::EdgeGNSS(v0_pose, last_gnss.utm_pose_);
  edge_gnss0->setInformation(Mat6d::Identity() * 1e2);
  optimizer.addEdge(edge_gnss0);

  edge_gnss0->computeError();
  LOG(INFO) << "gnss0 init err: " << edge_gnss0->chi2();

  auto edge_gnss1 = new sad::EdgeGNSS(v1_pose, this_gnss.utm_pose_);
  edge_gnss1->setInformation(Mat6d::Identity() * 1e2);
  optimizer.addEdge(edge_gnss1);

  edge_gnss1->computeError();
  LOG(INFO) << "gnss1 init err: " << edge_gnss1->chi2();

  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);

  sad::NavStated corr_state(this_state.timestamp_, v1_pose->estimate().so3(),
                            v1_pose->estimate().translation(),
                            v1_vel->estimate(), v1_bg->estimate(),
                            v1_ba->estimate());
  LOG(INFO) << "corr state in opt: " << corr_state;

  // Get results and report error statistics
  LOG(INFO) << "chi2/error: ";
  LOG(INFO) << "preintegration: " << edge_inertial->chi2() << "/"
            << edge_inertial->error().transpose();
  LOG(INFO) << "gnss0: " << edge_gnss0->chi2() << ", "
            << edge_gnss0->error().transpose();
  LOG(INFO) << "gnss1: " << edge_gnss1->chi2() << ", "
            << edge_gnss1->error().transpose();
  LOG(INFO) << "bias: " << edge_gyro_rw->chi2() << "/"
            << edge_acc_rw->error().transpose();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}