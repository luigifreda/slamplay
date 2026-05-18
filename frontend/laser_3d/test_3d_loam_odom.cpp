#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/io/io_utils.h"
#include "ad/laser_3d/loam_like/loam_like_odom.h"
#include "ad/timer/timer.h"

#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/wxb/test1.bag",
              "path to wxb bag");
DEFINE_string(topic, "/velodyne_packets_1", "topic of lidar packets");
DEFINE_bool(display_map, true, "display map?");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string resultsPathDir = resultsDir + "/ad/laser_3d/loam_odom";
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }

  // Test LOAM-like odometry performance
  sad::LoamLikeOdom::Options options;
  options.display_realtime_cloud_ = FLAGS_display_map;
  sad::LoamLikeOdom lo(options);

  LOG(INFO) << "using topic: " << FLAGS_topic;
  sad::RosbagIO bag_io(fLS::FLAGS_bag_path);
  bag_io
      .AddVelodyneHandle(FLAGS_topic,
                         [&](sad::FullCloudPtr cloud) -> bool {
                           sad::common::Timer::Evaluate(
                               [&]() { lo.ProcessPointCloud(cloud); },
                               "Loam-like odom");
                           return true;
                         })
      .Go();

  lo.SaveMap(resultsPathDir + "/loam_map.pcd");

  sad::common::Timer::PrintAll();
  LOG(INFO) << "done.";

  return 0;
}
