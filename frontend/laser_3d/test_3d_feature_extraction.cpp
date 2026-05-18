#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/io/io_utils.h"
#include "ad/laser_3d/loam_like/feature_extraction.h"

#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/timer/timer.h"

#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

/// VLP-16 data is needed here; using wxb dataset
DEFINE_string(bag_path, dataDir + "/ad/datasets/wxb/test1.bag",
              "path to wxb bag");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Test corner and surface point extraction
  sad::FeatureExtraction feature_extraction;
  std::string resultsPathDir = resultsDir + "/ad/laser_3d/feature_extraction";
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::system(("rm -rf " + resultsPathDir + "/*.pcd").c_str());

  int frame_count = 0;

  sad::RosbagIO bag_io(fLS::FLAGS_bag_path);
  bag_io
      .AddVelodyneHandle(
          "/velodyne_packets_1",
          [&](sad::FullCloudPtr cloud) -> bool {
            sad::CloudPtr pcd_corner(new sad::PointCloudType),
                pcd_surf(new sad::PointCloudType);
            sad::common::Timer::Evaluate(
                [&]() {
                  feature_extraction.Extract(cloud, pcd_corner, pcd_surf);
                },
                "Feature Extraction");
            LOG(INFO) << "original pts:" << cloud->size()
                      << ", corners: " << pcd_corner->size()
                      << ", surf: " << pcd_surf->size();
            sad::SaveCloudToFile(resultsPathDir + "/corner" +
                                     std::to_string(frame_count) + ".pcd",
                                 *pcd_corner);
            sad::SaveCloudToFile(resultsPathDir + "/surf" +
                                     std::to_string(frame_count) + ".pcd",
                                 *pcd_surf);
            frame_count++;
            return true;
          })
      .Go();

  sad::common::Timer::PrintAll();
  LOG(INFO) << "done.";

  return 0;
}
