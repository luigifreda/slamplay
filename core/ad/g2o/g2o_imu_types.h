#ifndef SLAM_IN_AUTO_DRIVING_CH4_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_CH4_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>

#include "ad/common/eigen_types.h"
#include "ad/imu/imu_preintegration.h"

namespace sad {

/// Vertices and edges related to preintegration
/**
 * Preintegration edge
 * Connects 6 vertices: previous pose, v, bg, ba, and next pose, v
 * The measurement is 9-dimensional, i.e. the preintegration residual, ordered
 * as R, v, p The information matrix is obtained from the preintegration class
 * and computed in the constructor
 */
class EdgeInertial : public g2o::BaseMultiEdge<9, Vec9d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * The constructor requires a preintegration object
   * @param preinteg  Pointer to the preintegration object
   * @param gravity   Gravity vector
   * @param weight    Weight
   */
  EdgeInertial(std::shared_ptr<IMUPreintegration> preinteg,
               const Vec3d &gravity, double weight = 1.0);

  bool read(std::istream &is) override { return false; }
  bool write(std::ostream &os) const override { return false; }

  void computeError() override;
  void linearizeOplus() override;

  Eigen::Matrix<double, 24, 24> GetHessian() {
    linearizeOplus();
    Eigen::Matrix<double, 9, 24> J;
    J.block<9, 6>(0, 0) = _jacobianOplus[0];
    J.block<9, 3>(0, 6) = _jacobianOplus[1];
    J.block<9, 3>(0, 9) = _jacobianOplus[2];
    J.block<9, 3>(0, 12) = _jacobianOplus[3];
    J.block<9, 6>(0, 15) = _jacobianOplus[4];
    J.block<9, 3>(0, 21) = _jacobianOplus[5];
    return J.transpose() * information() * J;
  }

private:
  const double dt_;
  std::shared_ptr<IMUPreintegration> preint_ = nullptr;
  Vec3d grav_;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
