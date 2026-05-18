//
// Created by xiang on 2022/3/15.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/io/io_utils.h"
#include "ad/laser_2d/lidar_2d_utils.h"
#include "ad/laser_2d/likelihood_field.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/2dmapping/floor1.bag",
              "rosbag path");
DEFINE_string(method, "gauss-newton", "gauss-newton/g2o");

/// Test 2D likelihood field ICP

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
  Scan2d::Ptr last_scan = nullptr, current_scan = nullptr;

  /// We align the previous scan with the current scan
  rosbag_io
      .AddScan2DHandle(
          "/pavo_scan_bottom",
          [&](Scan2d::Ptr scan) {
            sad::LikelihoodField lf;
            current_scan = scan;
            SE2 pose;

            if (last_scan == nullptr) {
              last_scan = current_scan;
              return true;
            }

            lf.SetTargetScan(last_scan);
            lf.SetSourceScan(current_scan);

            if (FLAGS_method == "gauss-newton") {
              lf.AlignGaussNewton(pose);
            } else if (FLAGS_method == "g2o") {
              lf.AlignG2O(pose);
            }

            LOG(INFO) << "aligned pose: " << pose.translation().transpose()
                      << ", " << pose.so2().log();

            cv::Mat image;
            sad::Visualize2DScan(last_scan, SE2(), image,
                                 Vec3b(255, 0, 0)); // target in blue
            sad::Visualize2DScan(current_scan, pose, image,
                                 Vec3b(0, 0, 255)); // source in red
            cv::imshow("scan", image);

            /// Draw the target and its field function
            cv::Mat field_image = lf.GetFieldImage();
            sad::Visualize2DScan(last_scan, SE2(), field_image,
                                 Vec3b(255, 0, 0), 1000,
                                 20.0); // target in blue
            cv::imshow("field", field_image);
            cv::waitKey(10);

            last_scan = current_scan;
            return true;
          })
      .Go();

  return 0;
}