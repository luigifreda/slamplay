//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "ad/io/io_utils.h"
#include "lio_slam/localization_fusion.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(config_yaml, configDir + "/lio_slam/mapping.yaml", "Config file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::LocalizationFusion fusion(FLAGS_config_yaml);
  if (!fusion.Init()) {
    return -1;
  }

  auto yaml = YAML::LoadFile(FLAGS_config_yaml);
  auto bag_path = yaml["bag_path"].as<std::string>();
  sad::RosbagIO rosbag_io(bag_path, sad::DatasetType::NCLT);

  /// Feed RTK, lidar, and IMU messages to fusion
  rosbag_io
      .AddAutoRTKHandle([&fusion](GNSSPtr gnss) {
        fusion.ProcessRTK(gnss);
        return true;
      })
      .AddAutoPointCloudHandle(
          [&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            fusion.ProcessPointCloud(cloud);
            return true;
          })
      .AddImuHandle([&](IMUPtr imu) {
        fusion.ProcessIMU(imu);
        return true;
      })
      .Go();

  LOG(INFO) << "done.";
  sleep(10);
  return 0;
}