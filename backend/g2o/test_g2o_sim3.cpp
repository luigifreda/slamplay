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
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/types/slam3d/vertex_se3.h"

using namespace std;
using namespace g2o;

extern "C" void G2O_FACTORY_EXPORT g2o_type_VertexSE3(void);

// Convert SE3 Vertex to Sim3 Vertex
void ToVertexSim3(const g2o::VertexSE3& v_se3,
                  g2o::VertexSim3Expmap* const v_sim3) {
  Eigen::Isometry3d se3 = v_se3.estimate().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();
  g2o::Sim3 sim3(r, t, 1.0);

  v_sim3->setEstimate(sim3);
}

// Convert Sim3 Vertex to SE3 Vertex
void ToVertexSE3(const g2o::VertexSim3Expmap& v_sim3,
                 g2o::VertexSE3* const v_se3) {
  g2o::Sim3 sim3 = v_sim3.estimate().inverse();
  Eigen::Matrix3d r = sim3.rotation().toRotationMatrix();
  Eigen::Vector3d t = sim3.translation();
  Eigen::Isometry3d se3;
  se3 = r;
  se3.translation() = t;

  v_se3->setEstimate(se3);
}

// Converte EdgeSE3 to EdgeSim3
void ToEdgeSim3(const g2o::EdgeSE3& e_se3, g2o::EdgeSim3* const e_sim3) {
  Eigen::Isometry3d se3 = e_se3.measurement().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();
  g2o::Sim3 sim3(r, t, 1.0);

  e_sim3->setMeasurement(sim3);
}

// Using VertexSim3 and EdgeSim3 is the core of this example.
// This example optimize the data created by create_sphere.
// Because the data is recore by VertexSE3 and EdgeSE3, SE3 is used for
// interface and Sim is used for optimization.
// g2o_viewer is available to the result.

int main(int argc, char** argv) 
{
  g2o_type_VertexSE3();
  if (argc != 2) {
    cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
    return 1;
  }
  ifstream fin(argv[1]);
  if (!fin) {
    cout << "file " << argv[1] << " does not exist." << endl;
    return 1;
  }

  //  define the optimizer
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 7>> BlockSolverType;
  typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>
      LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  // Load and Save in SE3
  g2o::SparseOptimizer interface;
  interface.load(argv[1]);

  // Convert all vertices
  for (auto& tmp : interface.vertices()) {
    const int& id = tmp.first;
    g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(tmp.second);
    g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
    v_sim3->setId(id);
    v_sim3->setMarginalized(false);

    ToVertexSim3(*v_se3, v_sim3);
    optimizer.addVertex(v_sim3);
    if (id == 0) {
      v_sim3->setFixed(true);
    }
  }

  // Convert all edges
  int edge_index = 0;
  for (auto& tmp : interface.edges()) {
    g2o::EdgeSE3* e_se3 = static_cast<g2o::EdgeSE3*>(tmp);
    int idx0 = e_se3->vertex(0)->id();
    int idx1 = e_se3->vertex(1)->id();
    g2o::EdgeSim3* e_sim3 = new g2o::EdgeSim3();

    ToEdgeSim3(*e_se3, e_sim3);
    e_sim3->setId(edge_index++);
    e_sim3->setVertex(0, optimizer.vertices()[idx0]);
    e_sim3->setVertex(1, optimizer.vertices()[idx1]);
    e_sim3->information() = Eigen::Matrix<double, 7, 7>::Identity();

    optimizer.addEdge(e_sim3);
  }

  cout << "optimizing ..." << endl;
  optimizer.initializeOptimization();
  optimizer.optimize(30);

  cout << "saving optimization results in VertexSE3..." << endl;
  auto vertices_sim3 = optimizer.vertices();
  auto vertices_se3 = interface.vertices();

  for (auto& tmp : vertices_sim3) {
    const int& id = tmp.first;
    g2o::VertexSim3Expmap* v_sim3 =
        static_cast<g2o::VertexSim3Expmap*>(tmp.second);
    g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(vertices_se3[id]);

    ToVertexSE3(*v_sim3, v_se3);
  }

  interface.save("result.g2o");
  return 0;
}