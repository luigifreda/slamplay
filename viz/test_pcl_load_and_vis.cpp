#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "macros.h"

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(pcd_path, dataDir + "/ad/pcd/map_example.pcd",
              "point cloud file path");

/// This program displays a single point cloud, demonstrating basic PCL usage.
/// It simply calls PCL's visualization library, similar to pcl_viewer.
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_pcd_path.empty()) {
    LOG(ERROR) << "pcd path is empty";
    return -1;
  }

  // Load point cloud
  PointCloudType::Ptr cloud(new PointCloudType);
  pcl::io::loadPCDFile(FLAGS_pcd_path, *cloud);

  if (cloud->empty()) {
    LOG(ERROR) << "cannot load cloud file";
    return -1;
  }

  LOG(INFO) << "cloud points: " << cloud->size();

  // visualize
  pcl::visualization::PCLVisualizer viewer("cloud viewer");
  pcl::visualization::PointCloudColorHandlerGenericField<PointType> handle(
      cloud, "z"); // color by height
  viewer.addPointCloud<PointType>(cloud, handle);
  viewer.spin();

  return 0;
}
