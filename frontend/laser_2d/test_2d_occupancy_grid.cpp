//
// Created by xiang on 2022/3/15.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/common/sys_utils.h"
#include "ad/io/io_utils.h"
#include "ad/laser_2d/frame.h"
#include "ad/laser_2d/lidar_2d_utils.h"
#include "ad/laser_2d/occupancy_map.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/2dmapping/floor1.bag",
              "rosbag path");
DEFINE_string(method, "model/bresenham", "filling algorithm: model/bresenham");

/// Test occupancy grid generation

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);

  /// Test whether the occupancy grid generated from a single scan is correct
  rosbag_io
      .AddScan2DHandle(
          "/pavo_scan_bottom",
          [&](Scan2d::Ptr scan) {
            sad::OccupancyMap oc_map;
            if (FLAGS_method == "model") {
              sad::evaluate_and_call(
                  [&]() {
                    oc_map.AddLidarFrame(
                        std::make_shared<sad::Frame>(scan),
                        sad::OccupancyMap::GridMethod::MODEL_POINTS);
                  },
                  "Occupancy with model points");
            } else {
              sad::evaluate_and_call(
                  [&]() {
                    oc_map.AddLidarFrame(
                        std::make_shared<sad::Frame>(scan),
                        sad::OccupancyMap::GridMethod::BRESENHAM);
                  },
                  "Occupancy with bresenham");
            }
            cv::imshow("occupancy map", oc_map.GetOccupancyGridBlackWhite());
            cv::waitKey(10);
            return true;
          })
      .Go();

  return 0;
}