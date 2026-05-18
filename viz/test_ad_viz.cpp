
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>

#include "viz/ad/autonomous_driving_viz.h"

#include "macros.h"
#include <string>

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag
std::string first_pcd_file = dataDir + "/ad/pcd/first.pcd";

DEFINE_string(source, first_pcd_file, "The first point cloud path");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::ui::AutonomousDrivingViz ui;
  ui.Init();
  sad::CloudPtr source(new sad::PointCloudType);
  pcl::io::loadPCDFile(fLS::FLAGS_source, *source);

  LOG(INFO) << "set state";
  ui.UpdateScan(source, SE3());

  LOG(INFO) << "waiting";
  sleep(60);
  ui.Quit();

  return 0;
}