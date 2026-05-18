#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/io/io_utils.h"
#include "ad/laser_3d/ndt_3d.h"
#include "ad/laser_3d/ndt_lo_direct.h"
#include "ad/timer/timer.h"

#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

/// This program uses the ULHK dataset as an example
/// to test NDT-based Lidar Odometry.
/// When using PCL NDT, the NDT tree is rebuilt each time.
DEFINE_string(bag_path, dataDir + "/ad/datasets/ulhk/test2.bag",
              "path to rosbag");
DEFINE_string(dataset_type, "ULHK", "NCLT/ULHK/KITTI/WXB_3D"); // dataset type
DEFINE_bool(use_pcl_ndt, false, "use pcl ndt to align?");
DEFINE_bool(use_ndt_nearby_6, false, "use ndt nearby 6?");
DEFINE_bool(display_map, true, "display map?");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string resultsPathDir = resultsDir + "/ad/laser_3d/ndt_lo";
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }

  sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path,
                          sad::Str2DatasetType(FLAGS_dataset_type));

  sad::NDTLODirect::Options options;
  options.use_pcl_ndt_ = fLB::FLAGS_use_pcl_ndt;
  options.ndt3d_options_.nearby_type_ = FLAGS_use_ndt_nearby_6
                                            ? sad::Ndt3d::NearbyType::NEARBY6
                                            : sad::Ndt3d::NearbyType::CENTER;
  options.display_realtime_cloud_ = FLAGS_display_map;
  sad::NDTLODirect ndt_lo(options);

  rosbag_io
      .AddAutoPointCloudHandle(
          [&ndt_lo](sensor_msgs::PointCloud2::Ptr msg) -> bool {
            sad::common::Timer::Evaluate(
                [&]() {
                  SE3 pose;
                  ndt_lo.AddCloud(
                      sad::VoxelCloud(sad::PointCloud2ToCloudPtr(msg)), pose);
                },
                "NDT registration");
            return true;
          })
      .Go();

  if (FLAGS_display_map) {
    // Save the map
    ndt_lo.SaveMap(resultsPathDir + "/map.pcd");
  }

  sad::common::Timer::PrintAll();
  LOG(INFO) << "done.";

  return 0;
}
