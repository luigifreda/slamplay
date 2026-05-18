#include "imu_preintegration.h"
#include <glog/logging.h>

namespace sad {

IMUPreintegration::IMUPreintegration(Options options) {
    bg_ = options.init_bg_;
    ba_ = options.init_ba_;
    const float ng2 = options.noise_gyro_ * options.noise_gyro_;
    const float na2 = options.noise_acce_ * options.noise_acce_;
    noise_gyro_acce_.diagonal() << ng2, ng2, ng2, na2, na2, na2;
}

void IMUPreintegration::Integrate(const IMU &imu, double dt) {
    // Measurements after removing the bias
    Vec3d gyr = imu.gyro_ - bg_;  // Gyro
    Vec3d acc = imu.acce_ - ba_;  // Accelerometer

    // Update dv and dp, see (4.13) and (4.16)
    dp_ = dp_ + dv_ * dt + 0.5f * dR_.matrix() * acc * dt * dt;
    dv_ = dv_ + dR_ * acc * dt;

    // Do not update dR yet, because A and B still need the current dR

    // Jacobian coefficients of the motion equations, matrices A and B, see (4.29)
    // The remaining two terms are handled below
    Eigen::Matrix<double, 9, 9> A;
    A.setIdentity();
    Eigen::Matrix<double, 9, 6> B;
    B.setZero();

    Mat3d acc_hat = SO3::hat(acc);
    double dt2 = dt * dt;

    // NOTE: the upper-left blocks of A and B differ slightly from the formula
    A.block<3, 3>(3, 0) = -dR_.matrix() * dt * acc_hat;
    A.block<3, 3>(6, 0) = -0.5f * dR_.matrix() * acc_hat * dt2;
    A.block<3, 3>(6, 3) = dt * Mat3d::Identity();

    B.block<3, 3>(3, 3) = dR_.matrix() * dt;
    B.block<3, 3>(6, 3) = 0.5f * dR_.matrix() * dt2;

    // Update each Jacobian, see Eq. (4.39)
    dP_dba_ = dP_dba_ + dV_dba_ * dt - 0.5f * dR_.matrix() * dt2;                      // (4.39d)
    dP_dbg_ = dP_dbg_ + dV_dbg_ * dt - 0.5f * dR_.matrix() * dt2 * acc_hat * dR_dbg_;  // (4.39e)
    dV_dba_ = dV_dba_ - dR_.matrix() * dt;                                             // (4.39b)
    dV_dbg_ = dV_dbg_ - dR_.matrix() * dt * acc_hat * dR_dbg_;                         // (4.39c)

    // Rotation part
    Vec3d omega = gyr * dt;         // Rotation increment
    Mat3d rightJ = SO3::jr(omega);  // Right Jacobian
    SO3 deltaR = SO3::exp(omega);   // Rotation after exp
    dR_ = dR_ * deltaR;             // (4.9)

    A.block<3, 3>(0, 0) = deltaR.matrix().transpose();
    B.block<3, 3>(0, 0) = rightJ * dt;

    // Update the noise term
    cov_ = A * cov_ * A.transpose() + B * noise_gyro_acce_ * B.transpose();

    // Update dR_dbg
    dR_dbg_ = deltaR.matrix().transpose() * dR_dbg_ - rightJ * dt;  // (4.39a)

    // Increment integrated time
    dt_ += dt;
}

SO3 IMUPreintegration::GetDeltaRotation(const Vec3d &bg) { return dR_ * SO3::exp(dR_dbg_ * (bg - bg_)); }

Vec3d IMUPreintegration::GetDeltaVelocity(const Vec3d &bg, const Vec3d &ba) {
    return dv_ + dV_dbg_ * (bg - bg_) + dV_dba_ * (ba - ba_);
}

Vec3d IMUPreintegration::GetDeltaPosition(const Vec3d &bg, const Vec3d &ba) {
    return dp_ + dP_dbg_ * (bg - bg_) + dP_dba_ * (ba - ba_);
}

NavStated IMUPreintegration::Predict(const sad::NavStated &start, const Vec3d &grav) const {
    SO3 Rj = start.R_ * dR_;
    Vec3d vj = start.R_ * dv_ + start.v_ + grav * dt_;
    Vec3d pj = start.R_ * dp_ + start.p_ + start.v_ * dt_ + 0.5f * grav * dt_ * dt_;

    auto state = NavStated(start.timestamp_ + dt_, Rj, pj, vj);
    state.bg_ = bg_;
    state.ba_ = ba_;
    return state;
}

}  // namespace sad
