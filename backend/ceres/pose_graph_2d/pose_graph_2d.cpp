// An example of solving a graph-based formulation of Simultaneous Localization
// and Mapping (SLAM). It reads a 2D pose graph problem definition file in the
// g2o format, formulates and solves the Ceres optimization problem, and outputs
// the original and optimized poses to file for plotting.

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "angle_manifold.h"
#include "ceres/ceres.h"
#include "g2o/read_g2o.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_2d_error_term.h"
#include "types.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
std::string g2o_file = dataDir + "/g2o/input_M3500_g2o.g2o";

DEFINE_string(input, "", "The pose graph definition filename in g2o format.");

namespace ceres {
namespace examples {
namespace {

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const std::vector<Constraint2d>& constraints,
                              std::map<int, Pose2d>* poses,
                              ceres::Problem* problem) {
    CHECK(poses != nullptr);
    CHECK(problem != nullptr);
    if (constraints.empty()) {
        LOG(INFO) << "No constraints, no problem to optimize.";
        return;
    }

    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* angle_manifold = AngleManifold::Create();

    for (const auto& constraint : constraints) {
        auto pose_begin_iter = poses->find(constraint.id_begin);
        CHECK(pose_begin_iter != poses->end())
            << "Pose with ID: " << constraint.id_begin << " not found.";
        auto pose_end_iter = poses->find(constraint.id_end);
        CHECK(pose_end_iter != poses->end())
            << "Pose with ID: " << constraint.id_end << " not found.";

        const Eigen::Matrix3d sqrt_information =
            constraint.information.llt().matrixL();
        // Ceres will take ownership of the pointer.
        ceres::CostFunction* cost_function = PoseGraph2dErrorTerm::Create(
            constraint.x, constraint.y, constraint.yaw_radians, sqrt_information);
        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  &pose_begin_iter->second.x,
                                  &pose_begin_iter->second.y,
                                  &pose_begin_iter->second.yaw_radians,
                                  &pose_end_iter->second.x,
                                  &pose_end_iter->second.y,
                                  &pose_end_iter->second.yaw_radians);

        problem->SetManifold(&pose_begin_iter->second.yaw_radians, angle_manifold);
        problem->SetManifold(&pose_end_iter->second.yaw_radians, angle_manifold);
    }

    // The pose graph optimization problem has three DOFs that are not fully
    // constrained. This is typically referred to as gauge freedom. You can apply
    // a rigid body transformation to all the nodes and the optimization problem
    // will still have the exact same cost. The Levenberg-Marquardt algorithm has
    // internal damping which mitigate this issue, but it is better to properly
    // constrain the gauge freedom. This can be done by setting one of the poses
    // as constant so the optimizer cannot change it.
    auto pose_start_iter = poses->begin();
    CHECK(pose_start_iter != poses->end()) << "There are no poses.";
    problem->SetParameterBlockConstant(&pose_start_iter->second.x);
    problem->SetParameterBlockConstant(&pose_start_iter->second.y);
    problem->SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
    CHECK(problem != nullptr);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);

    std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
}

// Output the poses to the file with format: ID x y yaw_radians.
bool OutputPoses(const std::string& filename,
                 const std::map<int, Pose2d>& poses) {
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
        std::cerr << "Error opening the file: " << filename << '\n';
        return false;
    }
    for (const auto& pair : poses) {
        outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
                << pair.second.yaw_radians << '\n';
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

    std::map<int, ceres::examples::Pose2d> poses;
    std::vector<ceres::examples::Constraint2d> constraints;

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
