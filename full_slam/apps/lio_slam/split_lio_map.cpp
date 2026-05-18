//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_cloud_utils.h"

#include "lio_slam/keyframe.h"
#include "lio_slam/map_utils.h"

#include "macros.h"
#include <filesystem>

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

std::string resultsLioMapPath =
    sad::resultsLioMappingPath; // where the keyframes are stored
std::string resultsLioMapDataPath =
    sad::resultsLioMappingDataPath; // where the map tiles are stored

DEFINE_string(map_path, resultsLioMapPath, "Directory to export data");
DEFINE_double(voxel_size, 0.1, "Exported map resolution");
DEFINE_double(grid_size, 100.0, "Grid size");

// Split the LIO map into smaller grids for better visualization
// The map is split into a grid of 100x100 cells, each cell is 100 meters wide.
// The map is saved as a set of PCD files, one for each cell.
// The index of the cells is saved in a text file.
// The index is the row and column of the cell in the grid.
// The PCD files are saved in the resultsLioMapDataPath directory.
// The index file is saved in the resultsLioMapDataPath directory.
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (!std::filesystem::exists(resultsLioMapPath)) {
    std::filesystem::create_directories(resultsLioMapPath);
  }

  if (!std::filesystem::exists(resultsLioMapDataPath)) {
    std::filesystem::create_directories(resultsLioMapDataPath);
  }

  int res = sad::splitLioMap(resultsLioMapPath, resultsLioMapDataPath,
                             FLAGS_voxel_size, FLAGS_grid_size);

  LOG(INFO) << "done.";
  if (res != 0) {
    LOG(ERROR) << "failed to split LIO map";
    return -1;
  }

  return 0;
}
