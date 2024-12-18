// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

///This program demonstrates the basic usage of sophus

int main(int argc, char **argv) 
{
  // Rotation matrix for 90 degrees along the Z axis
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  // or quaternion
  Quaterniond q(R);
  Sophus::SO3d SO3_R(R); // Sophus::SO3d can be constructed directly from the rotation matrix
  Sophus::SO3d SO3_q(q); // can also be constructed by quaternion
  // both are equivalent
  cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
  cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "they are equal" << endl;

  // Get its Lie algebra using the logarithmic map
  Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;
  // hat is vector to anti-symmetric matrix
  cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
  // Relatively, vee is antisymmetric to vector
  cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  // update of the incremental perturbation model
  Vector3d update_so3(1e-4, 0, 0);//Assume that the amount of updates is so much
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

  cout << "*******************************" << endl;

  // The operations on SE(3) are similar
  Vector3d t(1, 0, 0);//translate 1 along the X axis
  Sophus::SE3d SE3_Rt(R, t); // Construct SE(3) from R,t
  Sophus::SE3d SE3_qt(q, t); // Construct SE(3) from q,t
  cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
  cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;
  // Lie algebra se(3) is a six-dimensional vector, for convenience, first typedef it
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  // Observe the output, you will find that in Sophus, the translation of se(3) is in front, and the rotation is in the back.
  // Similarly, there are two operators hat and vee
  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

  //Finally, demonstrate the update
  Vector6d update_se3;//update amount
  update_se3.setZero();
  update_se3(0, 0) = 1e-4;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}
