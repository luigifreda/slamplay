#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

#include "json.h"
#include "macros.h"

using namespace std;
using namespace cv;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
string data_file = dataDir + "/rgbd2/points_and_matches.json";

typedef vector<Eigen::Vector2d> VecVector2d;
typedef vector<Eigen::Vector3d> VecVector3d;

void readDataFromJson(
    const std::string &filename,
    VecVector3d &points_3d,
    VecVector2d &points_2d,
    Mat &K);

// BA
void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

int main(int argc, char **argv) {
    if (argc == 2)
    {
        data_file = argv[1];
    } else if (argc != 2)
    {
        cout << "usage: " << argv[0] << " data" << endl;
    }

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    Mat K;

    readDataFromJson(data_file, pts_3d_eigen, pts_2d_eigen, K);

    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    auto t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}

void readDataFromJson(
    const std::string &filename,
    VecVector3d &pts_3d,
    VecVector2d &pts_2d,
    Mat &K) {
    std::ifstream i(filename);
    json j;
    i >> j;

    double fx = j["camera"]["fx"];
    double fy = j["camera"]["fy"];
    double cx = j["camera"]["cx"];
    double cy = j["camera"]["cy"];

    K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    pts_3d.clear();
    for (auto &p3d : j["points3d"])
    {
        pts_3d.emplace_back(p3d[0], p3d[1], p3d[2]);
    }

    pts_2d.clear();
    for (auto &p2d : j["points2d"])
    {
        pts_2d.emplace_back(p2d[0], p2d[1]);
    }

    assert(pts_3d.size() == pts_2d.size());
    std::cout << "read " << pts_3d.size() << " 3d points" << std::endl;
    std::cout << "read " << pts_2d.size() << " 2d points" << std::endl;
    std::cout << "read camera K: \n"
              << K << std::endl;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
            << -fx / Z,
            0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
            0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

   private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {
    // Build graph optimization, first set g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;            // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // linear solver type
    // Gradient descent method, you can choose from GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // graph model
    optimizer.setAlgorithm(solver);  // set up the solver
    optimizer.setVerbose(true);      // turn on debug output

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n"
         << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}
