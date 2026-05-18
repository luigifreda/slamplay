//
// Created by xiang on 2022/3/15.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/io/io_utils.h"
#include "ad/laser_2d/icp_2d.h"
#include "ad/laser_2d/lidar_2d_utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/2dmapping/floor1.bag",
              "rosbag path");
DEFINE_string(method, "point2point", "2D ICP method: point2point/point2plane");

/// Test reading 2D scans from rosbag and plotting the results
/// Select the method to use point-to-point or point-to-plane ICP
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
            current_scan = scan;

            if (last_scan == nullptr) {
              last_scan = current_scan;
              return true;
            }

            sad::Icp2d icp;
            icp.SetTarget(last_scan);
            icp.SetSource(current_scan);

            SE2 pose;
            if (fLS::FLAGS_method == "point2point") {
              icp.AlignGaussNewton(pose);
            } else if (fLS::FLAGS_method == "point2plane") {
              icp.AlignGaussNewtonPoint2Plane(pose);
            }

            cv::Mat image;
            sad::Visualize2DScan(last_scan, SE2(), image,
                                 Vec3b(255, 0, 0)); // target in blue
            sad::Visualize2DScan(current_scan, pose, image,
                                 Vec3b(0, 0, 255)); // source in red
            cv::imshow("scan", image);
            cv::waitKey(20);

            last_scan = current_scan;
            return true;
          })
      .Go();

  return 0;
}