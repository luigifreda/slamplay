#ifndef SLAM_IN_AUTO_DRIVING_COMMON_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_COMMON_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>

#include "ad/nav/nav_state.h"
#include "ad/pointcloud/point_types.h"

#include "ad/imu/imu_preintegration.h"
#include "g2o/core/robust_kernel_impl.h"

#include <glog/logging.h>

namespace sad {
/**
 * SO3+t pose with rotation first, 6-DoF; stored in a g2o::VertexSE3-like format
 * for g2o_viewer
 */
class VertexPose : public g2o::BaseVertex<6, SE3> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexPose() {}

  bool read(std::istream &is) override {
    double data[7];
    for (int i = 0; i < 7; i++) {
      is >> data[i];
    }
    setEstimate(SE3(Quatd(data[6], data[3], data[4], data[5]),
                    Vec3d(data[0], data[1], data[2])));
    return true;
  }

  bool write(std::ostream &os) const override {
    os << "VERTEX_SE3:QUAT ";
    os << id() << " ";
    Quatd q = _estimate.unit_quaternion();
    os << _estimate.translation().transpose() << " ";
    os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " "
       << q.coeffs()[3] << std::endl;
    return true;
  }

  virtual void setToOriginImpl() {}

  virtual void oplusImpl(const double *update_) {
    _estimate.so3() =
        _estimate.so3() *
        SO3::exp(Eigen::Map<const Vec3d>(&update_[0])); // Rotation part
    _estimate.translation() +=
        Eigen::Map<const Vec3d>(&update_[3]); // Translation part
    updateCache();
  }
};

/**
 * Velocity vertex, plain Vec3d
 */
class VertexVelocity : public g2o::BaseVertex<3, Vec3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexVelocity() {}

  virtual bool read(std::istream &is) { return false; }
  virtual bool write(std::ostream &os) const { return false; }

  virtual void setToOriginImpl() { _estimate.setZero(); }

  virtual void oplusImpl(const double *update_) {
    _estimate += Eigen::Map<const Vec3d>(update_);
  }
};

/**
 * Gyroscope bias vertex, also Vec3d, inherited from velocity vertex
 */
class VertexGyroBias : public VertexVelocity {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexGyroBias() {}
};

/**
 * Accelerometer bias vertex, Vec3d, also inherited from velocity vertex
 */
class VertexAccBias : public VertexVelocity {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexAccBias() {}
};

/**
 * Gyroscope random walk
 */
class EdgeGyroRW
    : public g2o::BaseBinaryEdge<3, Vec3d, VertexGyroBias, VertexGyroBias> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeGyroRW() {}

  virtual bool read(std::istream &is) { return false; }
  virtual bool write(std::ostream &os) const { return false; }

  void computeError() {
    const auto *VG1 = dynamic_cast<const VertexGyroBias *>(_vertices[0]);
    const auto *VG2 = dynamic_cast<const VertexGyroBias *>(_vertices[1]);
    _error = VG2->estimate() - VG1->estimate();
  }

  virtual void linearizeOplus() {
    _jacobianOplusXi = -Mat3d::Identity();
    _jacobianOplusXj.setIdentity();
  }

  Eigen::Matrix<double, 6, 6> GetHessian() {
    linearizeOplus();
    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = _jacobianOplusXi;
    J.block<3, 3>(0, 3) = _jacobianOplusXj;
    return J.transpose() * information() * J;
  }
};

/**
 * Accelerometer random walk
 */
class EdgeAccRW
    : public g2o::BaseBinaryEdge<3, Vec3d, VertexAccBias, VertexAccBias> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeAccRW() {}

  virtual bool read(std::istream &is) { return false; }
  virtual bool write(std::ostream &os) const { return false; }

  void computeError() {
    const auto *VA1 = dynamic_cast<const VertexAccBias *>(_vertices[0]);
    const auto *VA2 = dynamic_cast<const VertexAccBias *>(_vertices[1]);
    _error = VA2->estimate() - VA1->estimate();
  }

  virtual void linearizeOplus() {
    _jacobianOplusXi = -Mat3d::Identity();
    _jacobianOplusXj.setIdentity();
  }

  Eigen::Matrix<double, 6, 6> GetHessian() {
    linearizeOplus();
    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = _jacobianOplusXi;
    J.block<3, 3>(0, 3) = _jacobianOplusXj;
    return J.transpose() * information() * J;
  }
};

/**
 * Prior on IMU pvq and bias of the previous frame
 * info is provided externally from sliding-window marginalization
 *
 * Vertex order: pose, v, bg, ba
 * Residual order: R, p, v, bg, ba (15D)
 */
class EdgePriorPoseNavState : public g2o::BaseMultiEdge<15, Vec15d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgePriorPoseNavState(const NavStated &state, const Mat15d &info);

  virtual bool read(std::istream &is) { return false; }
  virtual bool write(std::ostream &os) const { return false; }

  void computeError();
  virtual void linearizeOplus();

  Eigen::Matrix<double, 15, 15> GetHessian() {
    linearizeOplus();
    Eigen::Matrix<double, 15, 15> J;
    J.block<15, 6>(0, 0) = _jacobianOplus[0];
    J.block<15, 3>(0, 6) = _jacobianOplus[1];
    J.block<15, 3>(0, 9) = _jacobianOplus[2];
    J.block<15, 3>(0, 12) = _jacobianOplus[3];
    return J.transpose() * information() * J;
  }

  NavStated state_;
};

/**
 * 6-DoF GNSS
 * Error order is rotation first, translation second
 */
class EdgeGNSS : public g2o::BaseUnaryEdge<6, SE3, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeGNSS() = default;
  EdgeGNSS(VertexPose *v, const SE3 &obs) {
    setVertex(0, v);
    setMeasurement(obs);
  }

  void computeError() override {
    VertexPose *v = (VertexPose *)_vertices[0];
    _error.head<3>() =
        (_measurement.so3().inverse() * v->estimate().so3()).log();
    _error.tail<3>() = v->estimate().translation() - _measurement.translation();
  };

  void linearizeOplus() override {
    VertexPose *v = (VertexPose *)_vertices[0];
    // jacobian 6x6
    _jacobianOplusXi.setZero();
    _jacobianOplusXi.block<3, 3>(0, 0) =
        (_measurement.so3().inverse() * v->estimate().so3()).jr_inv(); // dR/dR
    _jacobianOplusXi.block<3, 3>(3, 3) = Mat3d::Identity();            // dp/dp
  }

  Mat6d GetHessian() {
    linearizeOplus();
    return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
  }

  virtual bool read(std::istream &in) { return true; }
  virtual bool write(std::ostream &out) const { return true; }

private:
};

/**
 * Translation-only GNSS
 * Requires RTK extrinsic TBG to apply constraints correctly
 */
class EdgeGNSSTransOnly : public g2o::BaseUnaryEdge<3, Vec3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  /**
   * Specify pose vertex, RTK observation t_WG, and extrinsic TGB
   * @param v
   * @param obs
   */
  EdgeGNSSTransOnly(VertexPose *v, const Vec3d &obs, const SE3 &TBG)
      : TBG_(TBG) {
    setVertex(0, v);
    setMeasurement(obs);
  }

  void computeError() override {
    VertexPose *v = (VertexPose *)_vertices[0];
    _error = (v->estimate() * TBG_).translation() - _measurement;
  };

  // void linearizeOplus() override {
  //     VertexPose* v = (VertexPose*)_vertices[0];
  //     // jacobian 6x6
  //     _jacobianOplusXi.setZero();
  //     _jacobianOplusXi.block<3, 3>(0, 0) = (_measurement.so3().inverse() *
  //     v->estimate().so3()).jr_inv();  // dR/dR _jacobianOplusXi.block<3,
  //     3>(3, 3) = Mat3d::Identity(); // dp/dp
  // }

  virtual bool read(std::istream &in) { return true; }
  virtual bool write(std::ostream &out) const { return true; }

private:
  SE3 TBG_;
};

/**
 * 6-DoF relative motion
 * Error order is translation first, rotation second
 * Observation: T12
 */
class EdgeRelativeMotion
    : public g2o::BaseBinaryEdge<6, SE3, VertexPose, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeRelativeMotion() = default;
  EdgeRelativeMotion(VertexPose *v1, VertexPose *v2, const SE3 &obs) {
    setVertex(0, v1);
    setVertex(1, v2);
    setMeasurement(obs);
  }

  void computeError() override {
    VertexPose *v1 = (VertexPose *)_vertices[0];
    VertexPose *v2 = (VertexPose *)_vertices[1];
    // SE3 T12 = v1->estimate().inverse() * v2->estimate();
    _error =
        (_measurement.inverse() * v1->estimate().inverse() * v2->estimate())
            .log();
  };

  virtual bool read(std::istream &is) override {
    double data[7];
    for (int i = 0; i < 7; i++) {
      is >> data[i];
    }
    Quatd q(data[6], data[3], data[4], data[5]);
    q.normalize();
    setMeasurement(SE3(q, Vec3d(data[0], data[1], data[2])));
    for (int i = 0; i < information().rows() && is.good(); i++) {
      for (int j = i; j < information().cols() && is.good(); j++) {
        is >> information()(i, j);
        if (i != j)
          information()(j, i) = information()(i, j);
      }
    }
    return true;
  }

  virtual bool write(std::ostream &os) const override {
    os << "EDGE_SE3:QUAT ";
    auto *v1 = static_cast<VertexPose *>(_vertices[0]);
    auto *v2 = static_cast<VertexPose *>(_vertices[1]);
    os << v1->id() << " " << v2->id() << " ";
    SE3 m = _measurement;
    Eigen::Quaterniond q = m.unit_quaternion();
    os << m.translation().transpose() << " ";
    os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " "
       << q.coeffs()[3] << " ";

    // information matrix
    for (int i = 0; i < information().rows(); i++) {
      for (int j = i; j < information().cols(); j++) {
        os << information()(i, j) << " ";
      }
    }
    os << std::endl;
    return true;
  }

private:
};

/**
 * 3D wheel-encoder observation edge
 * Wheel speed observes world velocity projected into the vehicle frame; assumes
 * no y/z velocity in 3D
 */
class EdgeEncoder3D : public g2o::BaseUnaryEdge<3, Vec3d, VertexVelocity> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeEncoder3D() = default;

  /**
   * Constructor needs velocity in world frame
   * @param v0
   * @param speed
   */
  EdgeEncoder3D(VertexVelocity *v0, const Vec3d &speed) {
    setVertex(0, v0);
    setMeasurement(speed);
  }

  void computeError() override {
    VertexVelocity *v0 = (VertexVelocity *)_vertices[0];
    _error = v0->estimate() - _measurement;
  }

  void linearizeOplus() override { _jacobianOplusXi.setIdentity(); }

  virtual bool read(std::istream &in) { return true; }
  virtual bool write(std::ostream &out) const { return true; }
};

/**
 * NDT error model
 * Residual is Rp+t-mu, and info is estimated internally by NDT
 */
class EdgeNDT : public g2o::BaseUnaryEdge<3, Vec3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeNDT() = default;

  /// Need to query NDT internal voxels; pass it in as a callable
  using QueryVoxelFunc =
      std::function<bool(const Vec3d &query_pt, Vec3d &mu, Mat3d &info)>;

  EdgeNDT(VertexPose *v0, const Vec3d &pt, QueryVoxelFunc func) {
    setVertex(0, v0);
    pt_ = pt;
    query_ = func;

    Vec3d q = v0->estimate().so3() * pt_ + v0->estimate().translation();
    if (query_(q, mu_, info_)) {
      setInformation(info_);
      valid_ = true;
    } else {
      valid_ = false;
    }
  }

  bool IsValid() const { return valid_; }

  Mat6d GetHessian() {
    linearizeOplus();
    return _jacobianOplusXi.transpose() * info_ * _jacobianOplusXi;
  }

  /// Residual computation
  void computeError() override {
    VertexPose *v0 = (VertexPose *)_vertices[0];
    Vec3d q = v0->estimate().so3() * pt_ + v0->estimate().translation();

    if (query_(q, mu_, info_)) {
      _error = q - mu_;
      setInformation(info_);
      valid_ = true;
    } else {
      valid_ = false;
      _error.setZero();
      setLevel(1);
    }
  }

  /// Linearization
  void linearizeOplus() override {
    if (valid_) {
      VertexPose *v0 = (VertexPose *)_vertices[0];
      SO3 R = v0->estimate().so3();

      _jacobianOplusXi.setZero();
      _jacobianOplusXi.block<3, 3>(0, 0) =
          -R.matrix() * SO3::hat(pt_);                        // w.r.t. R
      _jacobianOplusXi.block<3, 3>(0, 3) = Mat3d::Identity(); // w.r.t. p
    } else {
      _jacobianOplusXi.setZero();
    }
  }

  virtual bool read(std::istream &in) { return true; }
  virtual bool write(std::ostream &out) const { return true; }

private:
  QueryVoxelFunc query_;
  Vec3d pt_ = Vec3d::Zero();
  Vec3d mu_ = Vec3d::Zero();
  Mat3d info_ = Mat3d::Identity();
  bool valid_ = false;
};
} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
