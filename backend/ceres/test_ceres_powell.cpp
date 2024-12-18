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
// An example program that minimizes Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.
//
// From: Testing Unconstrained Optimization Software by Jorge J. More, Burton S.
// Garbow and Kenneth E. Hillstrom in ACM Transactions on Mathematical Software,
// Vol 7(1), March 1981.

#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


struct F1 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x2, T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = x1[0] + 10.0 * x2[0];
    return true;
  }
};
struct F2 {
  template <typename T>
  bool operator()(const T* const x3, const T* const x4, T* residual) const {
    // f2 = sqrt(5) (x3 - x4)
    residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
    return true;
  }
};
struct F3 {
  template <typename T>
  bool operator()(const T* const x2, const T* const x3, T* residual) const {
    // f3 = (x2 - 2 x3)^2
    residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
    return true;
  }
};
struct F4 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x4, T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

DEFINE_string(minimizer,
              "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");

int main(int argc, char** argv) 
{
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // intial value
  double x1 = 3.0;
  double x2 = -1.0;
  double x3 = 0.0;
  double x4 = 1.0;

  Problem problem;

  // Add residual terms to the problem using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  problem.AddResidualBlock(
      new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2); // output size, first param size, second param size
  problem.AddResidualBlock(
      new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4); // output size, first param size, second param size
  problem.AddResidualBlock(
      new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3); // output size, first param size, second param size
  problem.AddResidualBlock(
      new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4); // output size, first param size, second param size
  Solver::Options options;

  LOG_IF(FATAL, !ceres::StringToMinimizerType(CERES_GET_FLAG(FLAGS_minimizer), &options.minimizer_type))
      << "Invalid minimizer: " << CERES_GET_FLAG(FLAGS_minimizer)
      << ", valid options are: trust_region and line_search.";
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  // clang-format off
  std::cout << "Initial x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  // clang-format on

  // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  
  // clang-format off
  std::cout << "Final x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  // clang-format on

  return 0;
}