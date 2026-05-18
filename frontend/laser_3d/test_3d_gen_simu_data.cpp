#include "ad/laser_3d/gen_simu_data.h"
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "ad/pointcloud/point_cloud_utils.h"
#include "macros.h"

#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

int main(int argc, char **argv) {
  sad::GenSimuData gen;
  gen.Gen();

  std::string resultsPathDir = resultsDir + "/ad/laser_3d/gen_simu_data";
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::system(("rm -rf " + resultsPathDir + "/*.pcd").c_str());

  sad::SaveCloudToFile(resultsPathDir + "/sim_source.pcd", *gen.GetSource());
  sad::SaveCloudToFile(resultsPathDir + "/sim_target.pcd", *gen.GetTarget());

  SE3 T_target_source = gen.GetPose().inverse();
  LOG(INFO) << "gt pose: " << T_target_source.translation().transpose() << ", "
            << T_target_source.so3().unit_quaternion().coeffs().transpose();

  return 0;
}