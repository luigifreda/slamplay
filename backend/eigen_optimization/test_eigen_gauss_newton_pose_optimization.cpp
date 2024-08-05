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

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
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

    cout << "calling bundle adjustment by Gauss-Newton" << endl;
    Sophus::SE3d pose_gn;
    auto t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by Gauss-Newton cost time: " << time_used.count() << " seconds." << endl;

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

void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (size_t i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;
            cost += e.squaredNorm();

            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            H += J.transpose() * J;   // H =  J^T * sigma^-1 * J
            b += -J.transpose() * e;  // b = -J^T * sigma^-1 * e
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n"
         << pose.matrix() << endl;
}
