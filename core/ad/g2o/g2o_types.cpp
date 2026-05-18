//
// Created by xiang on 23-1-11.
//

#include "ad/g2o/g2o_types.h"

namespace sad {

EdgePriorPoseNavState::EdgePriorPoseNavState(const NavStated &state,
                                             const Mat15d &info) {
  resize(4);
  state_ = state;
  setInformation(info);
}

void EdgePriorPoseNavState::computeError() {
  auto *vp = dynamic_cast<const VertexPose *>(_vertices[0]);
  auto *vv = dynamic_cast<const VertexVelocity *>(_vertices[1]);
  auto *vg = dynamic_cast<const VertexGyroBias *>(_vertices[2]);
  auto *va = dynamic_cast<const VertexAccBias *>(_vertices[3]);

  const Vec3d er =
      SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();
  const Vec3d ep = vp->estimate().translation() - state_.p_;
  const Vec3d ev = vv->estimate() - state_.v_;
  const Vec3d ebg = vg->estimate() - state_.bg_;
  const Vec3d eba = va->estimate() - state_.ba_;

  _error << er, ep, ev, ebg, eba;
}

void EdgePriorPoseNavState::linearizeOplus() {
  const auto *vp = dynamic_cast<const VertexPose *>(_vertices[0]);
  const Vec3d er =
      SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();

  /// Note there are 3 index types: vertex index, residual index, and local
  /// vertex variable index
  _jacobianOplus[0].setZero();
  _jacobianOplus[0].block<3, 3>(0, 0) = SO3::jr_inv(er);   // dr/dr
  _jacobianOplus[0].block<3, 3>(3, 3) = Mat3d::Identity(); // dp/dp
  _jacobianOplus[1].setZero();
  _jacobianOplus[1].block<3, 3>(6, 0) = Mat3d::Identity(); // dv/dv
  _jacobianOplus[2].setZero();
  _jacobianOplus[2].block<3, 3>(9, 0) = Mat3d::Identity(); // dbg/dbg
  _jacobianOplus[3].setZero();
  _jacobianOplus[3].block<3, 3>(12, 0) = Mat3d::Identity(); // dba/dba
}

} // namespace sad