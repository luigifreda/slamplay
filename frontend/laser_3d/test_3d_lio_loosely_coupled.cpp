#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/common/sys_utils.h"
#include "ad/io/io_utils.h"
#include "ad/laser_3d/lio_loosely_coupled/lio_loosely_coupled.h"
#include "ad/timer/timer.h"

#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(bag_path, dataDir + "/ad/datasets/ulhk/test3.bag",
              "path to rosbag");
DEFINE_string(dataset_type, "ULHK", "NCLT/ULHK/UTBM/AVIA"); // dataset type
DEFINE_string(config, configDir + "/lio_slam/velodyne_ulhk.yaml",
              "path of config yaml"); // config file type
DEFINE_bool(display_map, true, "display map?");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path,
                          sad::Str2DatasetType(FLAGS_dataset_type));

  sad::LIOLooselyCoupled::Options options;
  options.with_ui_ = FLAGS_display_map;
  sad::LIOLooselyCoupled lm(options);
  lm.Init(FLAGS_config);

  rosbag_io
      .AddAutoPointCloudHandle(
          [&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            sad::common::Timer::Evaluate([&]() { lm.PCLCallBack(cloud); },
                                         "loosely lio");
            return true;
          })
      .AddLivoxHandle(
          [&](const livox_ros_driver::CustomMsg::ConstPtr &msg) -> bool {
            sad::common::Timer::Evaluate([&]() { lm.LivoxPCLCallBack(msg); },
                                         "loosely lio");
            return true;
          })
      .AddImuHandle([&](IMUPtr imu) {
        lm.IMUCallBack(imu);
        return true;
      })
      .Go();

  lm.Finish();
  sad::common::Timer::PrintAll();
  LOG(INFO) << "done.";

  return 0;
}
