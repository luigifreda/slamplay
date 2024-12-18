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
#include <stdint.h>

#include <iostream>
#include <random>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/icp/types_icp.h"

using namespace Eigen;
using namespace std;
using namespace g2o;

int main(int argc, char** argv) 
{
  double euc_noise = 0.01;  // noise in position, m
  //double outlier_ratio = 0.1;

  SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  // variable-size block solver
  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverX>(
          g2o::make_unique<LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));

  optimizer.setAlgorithm(solver);

  vector<Vector3d> true_points;
  for (size_t i = 0; i < 1000; ++i) {
    true_points.push_back(
        Vector3d((g2o::Sampler::uniformRand(0., 1.) - 0.5) * 3,
                 g2o::Sampler::uniformRand(0., 1.) - 0.5,
                 g2o::Sampler::uniformRand(0., 1.) + 10));
  }

  // set up two poses
  int vertex_id = 0;
  for (size_t i = 0; i < 2; ++i) {
    // set up rotation and translation for this node
    Vector3d t(0, 0, i);
    Quaterniond q;
    q.setIdentity();

    Eigen::Isometry3d cam;  // camera pose
    cam = q;
    cam.translation() = t;

    // set up node
    VertexSE3* vc = new VertexSE3();
    vc->setEstimate(cam);

    vc->setId(vertex_id);  // vertex id

    cerr << t.transpose() << " | " << q.coeffs().transpose() << endl;

    // set first cam pose fixed
    if (i == 0) vc->setFixed(true);

    // add to optimizer
    optimizer.addVertex(vc);

    vertex_id++;
  }

  // set up point matches
  for (size_t i = 0; i < true_points.size(); ++i) {
    // get two poses
    VertexSE3* vp0 =
        dynamic_cast<VertexSE3*>(optimizer.vertices().find(0)->second);
    VertexSE3* vp1 =
        dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second);

    // calculate the relative 3D position of the point
    Vector3d pt0, pt1;
    pt0 = vp0->estimate().inverse() * true_points[i];
    pt1 = vp1->estimate().inverse() * true_points[i];

    // add in noise
    pt0 += Vector3d(g2o::Sampler::gaussRand(0., euc_noise),
                    g2o::Sampler::gaussRand(0., euc_noise),
                    g2o::Sampler::gaussRand(0., euc_noise));

    pt1 += Vector3d(g2o::Sampler::gaussRand(0., euc_noise),
                    g2o::Sampler::gaussRand(0., euc_noise),
                    g2o::Sampler::gaussRand(0., euc_noise));

    // form edge, with normals in varioius positions
    Vector3d nm0, nm1;
    nm0 << 0, i, 1;
    nm1 << 0, i, 1;
    nm0.normalize();
    nm1.normalize();

    Edge_V_V_GICP* e  // new edge with correct cohort for caching
        = new Edge_V_V_GICP();

    e->setVertex(0, vp0);  // first viewpoint
    e->setVertex(1, vp1);  // second viewpoint

    EdgeGICP meas;
    meas.pos0 = pt0;
    meas.pos1 = pt1;
    meas.normal0 = nm0;
    meas.normal1 = nm1;

    e->setMeasurement(meas);
    //        e->inverseMeasurement().pos() = -kp;

    meas = e->measurement();
    // use this for point-plane
    e->information() = meas.prec0(0.01);

    // use this for point-point
    // e->information().setIdentity();

    // e->setRobustKernel(true);
    // e->setHuberWidth(0.01);

    optimizer.addEdge(e);
  }

  // move second cam off of its true position
  VertexSE3* vc =
      dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second);
  Eigen::Isometry3d cam = vc->estimate();
  cam.translation() = Vector3d(0, 0, 0.2);
  vc->setEstimate(cam);

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;

  optimizer.setVerbose(true);

  optimizer.optimize(5);

  cout << endl << "Second vertex should be near 0,0,1" << endl;
  cout << dynamic_cast<VertexSE3*>(optimizer.vertices().find(0)->second)
              ->estimate()
              .translation()
              .transpose()
       << endl;
  cout << dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second)
              ->estimate()
              .translation()
              .transpose()
       << endl;

  return 0; 
}