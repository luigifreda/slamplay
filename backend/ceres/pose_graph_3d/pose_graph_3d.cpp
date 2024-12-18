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
#include <fstream>
#include <iostream>
#include <string>

#include "ceres/ceres.h"
#include "g2o/read_g2o.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_3d_error_term.h"
#include "types.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
// std::string g2o_file = dataDir + "/g2o/input_parking-garage.g2o";
std::string g2o_file = dataDir + "/g2o/input_sphere_bignoise_vertex3.g2o";

DEFINE_string(input, "", "The pose graph definition filename in g2o format.");

namespace ceres {
namespace examples {
namespace {

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              MapOfPoses* poses,
                              ceres::Problem* problem) {
    CHECK(poses != nullptr);
    CHECK(problem != nullptr);
    if (constraints.empty()) {
        LOG(INFO) << "No constraints, no problem to optimize.";
        return;
    }

    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* quaternion_manifold = new EigenQuaternionManifold;

    for (const auto& constraint : constraints) {
        auto pose_begin_iter = poses->find(constraint.id_begin);
        CHECK(pose_begin_iter != poses->end())
            << "Pose with ID: " << constraint.id_begin << " not found.";
        auto pose_end_iter = poses->find(constraint.id_end);
        CHECK(pose_end_iter != poses->end())
            << "Pose with ID: " << constraint.id_end << " not found.";

        const Eigen::Matrix<double, 6, 6> sqrt_information =
            constraint.information.llt().matrixL();
        // Ceres will take ownership of the pointer.
        ceres::CostFunction* cost_function =
            PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  pose_begin_iter->second.p.data(),
                                  pose_begin_iter->second.q.coeffs().data(),
                                  pose_end_iter->second.p.data(),
                                  pose_end_iter->second.q.coeffs().data());

        problem->SetManifold(pose_begin_iter->second.q.coeffs().data(),
                             quaternion_manifold);
        problem->SetManifold(pose_end_iter->second.q.coeffs().data(),
                             quaternion_manifold);
    }

    // The pose graph optimization problem has six DOFs that are not fully
    // constrained. This is typically referred to as gauge freedom. You can apply
    // a rigid body transformation to all the nodes and the optimization problem
    // will still have the exact same cost. The Levenberg-Marquardt algorithm has
    // internal damping which mitigates this issue, but it is better to properly
    // constrain the gauge freedom. This can be done by setting one of the poses
    // as constant so the optimizer cannot change it.
    auto pose_start_iter = poses->begin();
    CHECK(pose_start_iter != poses->end()) << "There are no poses.";
    problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
    problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
    CHECK(problem != nullptr);

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);

    std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const MapOfPoses& poses) {
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
        LOG(ERROR) << "Error opening the file: " << filename;
        return false;
    }
    for (const auto& pair : poses) {
        outfile << pair.first << " " << pair.second.p.transpose() << " "
                << pair.second.q.x() << " " << pair.second.q.y() << " "
                << pair.second.q.z() << " " << pair.second.q.w() << '\n';
    }
    return true;
}

}  // namespace
}  // namespace examples
}  // namespace ceres

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

#if 0
  CHECK(FLAGS_input != "") << "Need to specify the filename to read.";
#else
    std::string FLAGS_input = g2o_file;
#endif

    ceres::examples::MapOfPoses poses;
    ceres::examples::VectorOfConstraints constraints;

    CHECK(ceres::examples::ReadG2oFile(FLAGS_input, &poses, &constraints))
        << "Error reading the file: " << FLAGS_input;

    std::cout << "Number of poses: " << poses.size() << '\n';
    std::cout << "Number of constraints: " << constraints.size() << '\n';

    CHECK(ceres::examples::OutputPoses("poses_original.txt", poses))
        << "Error outputting to poses_original.txt";

    ceres::Problem problem;
    ceres::examples::BuildOptimizationProblem(constraints, &poses, &problem);

    CHECK(ceres::examples::SolveOptimizationProblem(&problem))
        << "The solve was not successful, exiting.";

    CHECK(ceres::examples::OutputPoses("poses_optimized.txt", poses))
        << "Error outputting to poses_original.txt";

    return 0;
}
