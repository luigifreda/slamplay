//
// Created by xiang on 2022/3/15.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/io/io_utils.h"
#include "ad/laser_2d/lidar_2d_utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/2dmapping/floor1.bag",
              "rosbag path");

/// Test reading 2D scans from rosbag and plotting the results

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
  rosbag_io
      .AddScan2DHandle("/pavo_scan_bottom",
                       [](Scan2d::Ptr scan) {
                         cv::Mat image;
                         sad::Visualize2DScan(scan, SE2(), image,
                                              Vec3b(255, 0, 0));
                         cv::imshow("scan", image);
                         cv::waitKey(20);
                         return true;
                       })
      .Go();

  return 0;
}