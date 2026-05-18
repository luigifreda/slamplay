#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "lio_slam/map_utils.h"

#include "macros.h"

std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(map_data_path, sad::resultsLioMappingDataPath,
              "Split map tiles directory (map_index.txt + tile PCDs)");
DEFINE_string(dump_to, sad::resultsLioMappingDataPath,
              "Output directory for map_colored.pcd");

// Load split map tiles (output of splitLioMap), assign a distinct color per
// grid cell, and merge into a single RGBA point cloud. The map tiles are loaded
// from the map_data_path directory. The map tiles are merged into a single RGBA
// point cloud.
// NOTE: We need to run splitLioMap first to generate the map tiles.
int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  sad::UiCloudPtr global_cloud(new sad::UiPointCloudType);
  int res =
      sad::dumpLioSplitMap(FLAGS_map_data_path, FLAGS_dump_to, global_cloud);
  if (res != 0) {
    LOG(ERROR) << "failed to dump LIO map data";
    return -1;
  }

  if (global_cloud->empty()) {
    LOG(INFO) << "colored map is empty, nothing to visualize";
    return 0;
  }

  pcl::visualization::PCLVisualizer viewer("lio map data");
  viewer.addCoordinateSystem(10.0, "world");
  viewer.addPointCloud<sad::UiPointType>(global_cloud, "map");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "map");

  while (!viewer.wasStopped()) {
    viewer.spinOnce(5);
  }

  return 0;
}
