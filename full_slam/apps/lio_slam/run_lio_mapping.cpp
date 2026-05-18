//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "lio_slam/frontend.h"
#include "lio_slam/loop_closure.h"
#include "lio_slam/map_utils.h"
#include "lio_slam/optimization.h"

#include "viz/ad/pcl_map_viewer.h"

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_cloud_utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(config_yaml, configDir + "/lio_slam/mapping.yaml", "Config file");
DEFINE_double(voxel_size, 0.1, "Exported map resolution");
DEFINE_string(pose_source, "lidar", "Pose source: lidar/rtk/opti1/opti2");

// Run LIO mapping pipeline. All the steps are executed in sequence.
// 1. Run frontend
// 2. Run optimization stage 1
// 3. Run loop closure
// 4. Run optimization stage 2 (with loop closure constraints)
// NOTE: This is not a online SLAM but it runs all the steps of a typical SLAM
// pipeline.
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "running frontend";
  sad::Frontend frontend(FLAGS_config_yaml);
  if (!frontend.Init()) {
    LOG(ERROR) << "failed to init frontend.";
    return -1;
  }

  frontend.Run();

  LOG(INFO) << "running optimization stage 1";
  sad::Optimization opti(FLAGS_config_yaml);
  if (!opti.Init(1)) {
    LOG(ERROR) << "failed to init opti1.";
    return -1;
  }
  opti.Run();

  LOG(INFO) << "running loop closure";
  sad::LoopClosure lc(FLAGS_config_yaml);
  if (!lc.Init()) {
    LOG(ERROR) << "failed to init loop closure.";
    return -1;
  }
  lc.Run();

  LOG(INFO) << "running optimization stage 2";
  sad::Optimization opti2(FLAGS_config_yaml);
  if (!opti2.Init(2)) {
    LOG(ERROR) << "failed to init opti2.";
    return -1;
  }
  opti2.Run();

  LOG(INFO) << "done.";

  sad::CloudPtr global_cloud(new sad::PointCloudType);
  std::string map_path = sad::resultsLioMappingPath;
  std::string output_path = map_path;
  int res = sad::dumpLioMap(map_path, output_path, FLAGS_voxel_size,
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
