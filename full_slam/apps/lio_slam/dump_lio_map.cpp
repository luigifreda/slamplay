#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_cloud_utils.h"
#include "lio_slam/keyframe.h"
#include "lio_slam/map_utils.h"

#include "viz/ad/pcl_map_viewer.h"

#include "macros.h"
#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

std::string resultsLioMapPath = sad::resultsLioMappingPath;

DEFINE_double(voxel_size, 0.1, "Exported map resolution");
DEFINE_string(pose_source, "lidar", "Pose source: lidar/rtk/opti1/opti2");
DEFINE_string(map_path, resultsLioMapPath, "Map input path");
DEFINE_string(dump_to, resultsLioMapPath, "Output directory");

// Merge the LIO map into a single PCD file
// The keyframes are loaded from the keyframes.txt file.
// The keyframes are merged into a single point cloud.
// The point cloud is saved as a PCD file in the resultsLioMapPath directory.
// NOTE: We need to run splitLioMap first to generate the map tiles.
int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  sad::CloudPtr global_cloud(new sad::PointCloudType);
  int res = sad::dumpLioMap(FLAGS_map_path, FLAGS_dump_to, FLAGS_voxel_size,
                            FLAGS_pose_source, global_cloud);
  if (res != 0) {
    LOG(ERROR) << "failed to dump LIO map";
    return -1;
  }

  // load the built map and visualize it
  sad::PCLMapViewer viewer(FLAGS_voxel_size);
  viewer.SetPoseAndCloud(SE3(), global_cloud);
  viewer.Spin(5);

  return 0;
}
