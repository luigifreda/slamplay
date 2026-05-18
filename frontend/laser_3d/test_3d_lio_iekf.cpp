//
// Created by xiang on 22-11-10.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/common/sys_utils.h"
#include "ad/io/io_utils.h"
#include "ad/laser_3d/lio_iekf/lio_iekf.h"
#include "ad/timer/timer.h"

#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/nclt/20120115.bag",
              "path to rosbag");
DEFINE_string(dataset_type, "NCLT", "NCLT/ULHK/UTBM/AVIA"); // dataset type
DEFINE_string(config, configDir + "/lio_slam/velodyne_nclt.yaml",
              "path of config yaml"); // config file path
DEFINE_bool(display_map, true, "display map?");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path,
                          sad::Str2DatasetType(FLAGS_dataset_type));

  sad::LioIEKF lio;
  lio.Init(FLAGS_config);

  rosbag_io
      .AddAutoPointCloudHandle(
          [&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.PCLCallBack(cloud); },
                                         "IEKF lio");
            return true;
          })
      .AddLivoxHandle(
          [&](const livox_ros_driver::CustomMsg::ConstPtr &msg) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.LivoxPCLCallBack(msg); },
                                         "IEKF lio");
            return true;
          })
      .AddImuHandle([&](IMUPtr imu) {
        lio.IMUCallBack(imu);
        return true;
      })
      .Go();

  lio.Finish();
  sad::common::Timer::PrintAll();
  LOG(INFO) << "done.";

  return 0;
}
