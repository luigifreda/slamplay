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
// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/icp/types_icp.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#else
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#endif

using namespace Eigen;
using namespace std;

class Sample {
 public:
  static int uniform(int from, int to) {
    return static_cast<int>(g2o::Sampler::uniformRand(from, to));
  }
};

int main(int argc, const char* argv[]) 
{
  if (argc < 2) {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] "
            "[STRUCTURE_ONLY] [DENSE]"
         << endl;
    cout << endl;
    cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
    cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)"
         << endl;
    cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)"
         << endl;
    cout << "STRUCTURE_ONLY: performed structure-only BA to get better point "
            "initializations (0 or 1; default: 0==false)"
         << endl;
    cout << "DENSE: Use dense solver (0 or 1; default: 0==false)" << endl;
    cout << endl;
    cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to "
            "1==true."
         << endl;
    cout << endl;
    exit(0);
  }

  double PIXEL_NOISE = atof(argv[1]);

  double OUTLIER_RATIO = 0.0;

  if (argc > 2) {
    OUTLIER_RATIO = atof(argv[2]);
  }

  bool ROBUST_KERNEL = false;
  if (argc > 3) {
    ROBUST_KERNEL = atoi(argv[3]) != 0;
  }
  bool STRUCTURE_ONLY = false;
  if (argc > 4) {
    STRUCTURE_ONLY = atoi(argv[4]) != 0;
  }

  bool DENSE = false;
  if (argc > 5) {
    DENSE = atoi(argv[5]) != 0;
  }

  cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;
  cout << "OUTLIER_RATIO: " << OUTLIER_RATIO << endl;
  cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
  cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY << endl;
  cout << "DENSE: " << DENSE << endl;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
  if (DENSE) {
    linearSolver = g2o::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    cerr << "Using DENSE" << endl;
  } else {
#ifdef G2O_HAVE_CHOLMOD
    cerr << "Using CHOLMOD" << endl;
    linearSolver = g2o::make_unique<
        g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
#else
    linearSolver = g2o::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    cerr << "Using CSPARSE" << endl;
#endif
  }

  g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg(
          g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

  optimizer.setAlgorithm(solver);

  // set up 500 points
  vector<Vector3d> true_points;
  for (size_t i = 0; i < 500; ++i) {
    true_points.push_back(
        Vector3d((g2o::Sampler::uniformRand(0., 1.) - 0.5) * 3,
                 g2o::Sampler::uniformRand(0., 1.) - 0.5,
                 g2o::Sampler::uniformRand(0., 1.) + 10));
  }

  Vector2d focal_length(500, 500);     // pixels
  Vector2d principal_point(320, 240);  // 640x480 image
  double baseline = 0.075;             // 7.5 cm baseline

  vector<Eigen::Isometry3d, aligned_allocator<Eigen::Isometry3d>> true_poses;

  // set up camera params
  g2o::VertexSCam::setKcam(focal_length[0], focal_length[1], principal_point[0],
                           principal_point[1], baseline);

  // set up 5 vertices, first 2 fixed
  int vertex_id = 0;
  for (size_t i = 0; i < 5; ++i) {
    Vector3d trans(i * 0.04 - 1., 0, 0);

    Eigen::Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d pose;
    pose = q;
    pose.translation() = trans;

    g2o::VertexSCam* v_se3 = new g2o::VertexSCam();

    v_se3->setId(vertex_id);
    v_se3->setEstimate(pose);
    v_se3->setAll();  // set aux transforms

    if (i < 2) v_se3->setFixed(true);

    optimizer.addVertex(v_se3);
    true_poses.push_back(pose);
    vertex_id++;
  }

  int point_id = vertex_id;
  int point_num = 0;
  double sum_diff2 = 0;

  cout << endl;
  unordered_map<int, int> pointid_2_trueid;
  unordered_set<int> inliers;

  // add point projections to this vertex
  for (size_t i = 0; i < true_points.size(); ++i) {
    g2o::VertexPointXYZ* v_p = new g2o::VertexPointXYZ();

    v_p->setId(point_id);
    v_p->setMarginalized(true);
    v_p->setEstimate(true_points.at(i) +
                     Vector3d(g2o::Sampler::gaussRand(0., 1),
                              g2o::Sampler::gaussRand(0., 1),
                              g2o::Sampler::gaussRand(0., 1)));

    int num_obs = 0;

    for (size_t j = 0; j < true_poses.size(); ++j) {
      Vector3d z;
      dynamic_cast<g2o::VertexSCam*>(optimizer.vertices().find(j)->second)
          ->mapPoint(z, true_points.at(i));

      if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
        ++num_obs;
      }
    }

    if (num_obs >= 2) {
      optimizer.addVertex(v_p);

      bool inlier = true;
      for (size_t j = 0; j < true_poses.size(); ++j) {
        Vector3d z;
        dynamic_cast<g2o::VertexSCam*>(optimizer.vertices().find(j)->second)
            ->mapPoint(z, true_points.at(i));

        if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
          double sam = g2o::Sampler::uniformRand(0., 1.);
          if (sam < OUTLIER_RATIO) {
            z = Vector3d(Sample::uniform(64, 640), Sample::uniform(0, 480),
                         Sample::uniform(0, 64));  // disparity
            z(2) = z(0) - z(2);                    // px' now

            inlier = false;
          }

          z += Vector3d(g2o::Sampler::gaussRand(0., PIXEL_NOISE),
                        g2o::Sampler::gaussRand(0., PIXEL_NOISE),
                        g2o::Sampler::gaussRand(0., PIXEL_NOISE / 16.0));

          g2o::Edge_XYZ_VSC* e = new g2o::Edge_XYZ_VSC();

          e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);

          e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(
              optimizer.vertices().find(j)->second);

          e->setMeasurement(z);
          // e->inverseMeasurement() = -z;
          e->information() = Matrix3d::Identity();

          if (ROBUST_KERNEL) {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
          }

          optimizer.addEdge(e);
        }
      }

      if (inlier) {
        inliers.insert(point_id);
        Vector3d diff = v_p->estimate() - true_points[i];

        sum_diff2 += diff.dot(diff);
      }
      // else
      //   cout << "Point: " << point_id <<  "has at least one spurious
      //   observation" <<endl;

      pointid_2_trueid.insert(make_pair(point_id, i));

      ++point_id;
      ++point_num;
    }
  }

  cout << endl;
  optimizer.initializeOptimization();

  optimizer.setVerbose(true);

  if (STRUCTURE_ONLY) {
    cout << "Performing structure-only BA:" << endl;
    g2o::StructureOnlySolver<3> structure_only_ba;
    g2o::OptimizableGraph::VertexContainer points;
    for (g2o::OptimizableGraph::VertexIDMap::const_iterator it =
             optimizer.vertices().begin();
         it != optimizer.vertices().end(); ++it) {
      g2o::OptimizableGraph::Vertex* v =
          static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() == 3) points.push_back(v);
    }

    structure_only_ba.calc(points, 10);
  }

  cout << endl;
  cout << "Performing full BA:" << endl;
  optimizer.optimize(10);

  cout << endl;
  cout << "Point error before optimisation (inliers only): "
       << sqrt(sum_diff2 / inliers.size()) << endl;

  point_num = 0;
  sum_diff2 = 0;

  for (unordered_map<int, int>::iterator it = pointid_2_trueid.begin();
       it != pointid_2_trueid.end(); ++it) {
    g2o::HyperGraph::VertexIDMap::iterator v_it =
        optimizer.vertices().find(it->first);

    if (v_it == optimizer.vertices().end()) {
      cerr << "Vertex " << it->first << " not in graph!" << endl;
      exit(-1);
    }

    g2o::VertexPointXYZ* v_p = dynamic_cast<g2o::VertexPointXYZ*>(v_it->second);

    if (v_p == 0) {
      cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
      exit(-1);
    }

    Vector3d diff = v_p->estimate() - true_points[it->second];

    if (inliers.find(it->first) == inliers.end()) continue;

    sum_diff2 += diff.dot(diff);

    ++point_num;
  }

  cout << "Point error after optimisation (inliers only): "
       << sqrt(sum_diff2 / inliers.size()) << endl;
  cout << endl;

  return 0; 
}