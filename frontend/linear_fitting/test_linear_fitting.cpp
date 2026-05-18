#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"

DEFINE_int32(num_tested_points_plane, 10, "number of tested points in plane fitting");
DEFINE_int32(num_tested_points_line, 100, "number of tested points in line fitting");
DEFINE_double(noise_sigma, 0.01, "noise of generated samples");

void PlaneFittingTest();
void LineFittingTest();

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    LOG(INFO) << "testing plane fitting";
    PlaneFittingTest();

    LOG(INFO) << "testing line fitting";
    LineFittingTest();
}

void PlaneFittingTest() {
    Vec4d true_plane_coeffs(0.1, 0.2, 0.3, 0.4);
    true_plane_coeffs.normalize();

    std::vector<Vec3d> points;

    // Randomly generate simulated plane points
        cv::RNG rng;
    for (int i = 0; i < FLAGS_num_tested_points_plane; ++i) {
        // Generate a random point, compute the 4th dimension, add noise, then normalize
        Vec3d p(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
        double n4 = -p.dot(true_plane_coeffs.head<3>()) / true_plane_coeffs[3];
        p = p / (n4 + std::numeric_limits<double>::min());  // prevent division by zero
        p += Vec3d(rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma));

        points.emplace_back(p);

        // Verify the point lies on the plane
        LOG(INFO) << "res of p: " << p.dot(true_plane_coeffs.head<3>()) + true_plane_coeffs[3];
    }

    Vec4d estimated_plane_coeffs;
    if (sad::math::FitPlane(points, estimated_plane_coeffs)) {
        LOG(INFO) << "estimated coeffs: " << estimated_plane_coeffs.transpose()
                  << ", true: " << true_plane_coeffs.transpose();
    } else {
        LOG(INFO) << "plane fitting failed";
    }
}

void LineFittingTest() {
    // Ground truth line fitting parameters
    Vec3d true_line_origin(0.1, 0.2, 0.3);
    Vec3d true_line_dir(0.4, 0.5, 0.6);
    true_line_dir.normalize();

    // Randomly generate line points using the parametric equation
    std::vector<Vec3d> points;
    cv::RNG rng;
    for (int i = 0; i < fLI::FLAGS_num_tested_points_line; ++i) {
        double t = rng.uniform(-1.0, 1.0);
        Vec3d p = true_line_origin + true_line_dir * t;
        p += Vec3d(rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma));

        points.emplace_back(p);
    }

    Vec3d esti_origin, esti_dir;
    if (sad::math::FitLine(points, esti_origin, esti_dir)) {
        LOG(INFO) << "estimated origin: " << esti_origin.transpose() << ", true: " << true_line_origin.transpose();
        LOG(INFO) << "estimated dir: " << esti_dir.transpose() << ", true: " << true_line_dir.transpose();
    } else {
        LOG(INFO) << "line fitting failed";
    }
}
