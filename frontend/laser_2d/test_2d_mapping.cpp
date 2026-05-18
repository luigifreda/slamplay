//
// Created by xiang on 2022/3/15.
//
#include <filesystem>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/io/io_utils.h"
#include "ad/laser_2d/lidar_2d_utils.h"
#include "ad/laser_2d/mapping_2d.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/2dmapping/floor1.bag",
              "rosbag path");
DEFINE_bool(with_loop_closing, false, "whether to use loop closing");

/// Test 2D lidar SLAM

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
  sad::Mapping2D mapping;

  std::string resultsPathDir = resultsDir + "/ad/2dmapping";
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::system(("rm -rf " + resultsPathDir + "/*").c_str());

  LOG(INFO) << "Initializing mapping with loop closing: "
            << FLAGS_with_loop_closing;
  if (mapping.Init(FLAGS_with_loop_closing) == false) {
    return -1;
  }

  rosbag_io
      .AddScan2DHandle(
          "/pavo_scan_bottom",
          [&](Scan2d::Ptr scan) { return mapping.ProcessScan(scan); })
      .Go();
  cv::imwrite(resultsPathDir + "/global_map.png", mapping.ShowGlobalMap(2000));
  return 0;
}