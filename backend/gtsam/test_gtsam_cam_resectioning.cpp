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
/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    CameraResectioning.cpp
 * @brief   An example of gtsam for solving the camera resectioning problem
 * @author  Duy-Nguyen Ta
 * @date    Aug 23, 2011
 */

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <boost/make_shared.hpp>

using namespace gtsam;
using namespace gtsam::noiseModel;
using symbol_shorthand::X;

/**
 * Unary factor on the unknown pose, resulting from meauring the projection of
 * a known 3D point in the image
 */
class ResectioningFactor: public NoiseModelFactorN<Pose3> {
  typedef NoiseModelFactorN<Pose3> Base;

  Cal3_S2::shared_ptr K_; ///< camera's intrinsic parameters
  Point3 P_;              ///< 3D point on the calibration rig
  Point2 p_;              ///< 2D measurement of the 3D point

public:

  /// Construct factor given known point P and its projection p
  ResectioningFactor(const SharedNoiseModel& model, const Key& key,
      const Cal3_S2::shared_ptr& calib, const Point2& p, const Point3& P) :
      Base(model, key), K_(calib), P_(P), p_(p) {
  }

  /// evaluate the error
  Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H =
      boost::none) const override {
    PinholeCamera<Cal3_S2> camera(pose, *K_);
    return camera.project(P_, H, boost::none, boost::none) - p_;
  }
};

/*******************************************************************************
 * Camera: f = 1, Image: 100x100, center: 50, 50.0
 * Pose (ground truth): (Xw, -Yw, -Zw, [0,0,2.0]')
 * Known landmarks:
 *    3D Points: (10,10,0) (-10,10,0) (-10,-10,0) (10,-10,0)
 * Perfect measurements:
 *    2D Point:  (55,45)   (45,45)    (45,55)     (55,55)
 *******************************************************************************/
int main(int argc, char* argv[]) {
  /* read camera intrinsic parameters */
  Cal3_S2::shared_ptr calib(new Cal3_S2(1, 1, 0, 50, 50));

  /* 1. create graph */
  NonlinearFactorGraph graph;

  /* 2. add factors to the graph */
  // add measurement factors
  SharedDiagonal measurementNoise = Diagonal::Sigmas(Vector2(0.5, 0.5));
  boost::shared_ptr<ResectioningFactor> factor;
  graph.emplace_shared<ResectioningFactor>(measurementNoise, X(1), calib,
          Point2(55, 45), Point3(10, 10, 0));
  graph.emplace_shared<ResectioningFactor>(measurementNoise, X(1), calib,
          Point2(45, 45), Point3(-10, 10, 0));
  graph.emplace_shared<ResectioningFactor>(measurementNoise, X(1), calib,
          Point2(45, 55), Point3(-10, -10, 0));
  graph.emplace_shared<ResectioningFactor>(measurementNoise, X(1), calib,
          Point2(55, 55), Point3(10, -10, 0));

  /* 3. Create an initial estimate for the camera pose */
  Values initial;
  initial.insert(X(1),
      Pose3(Rot3(1, 0, 0, 0, -1, 0, 0, 0, -1), Point3(0, 0, 2)));

  /* 4. Optimize the graph using Levenberg-Marquardt*/
  Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
  result.print("Final result:\n");

  return 0;
}