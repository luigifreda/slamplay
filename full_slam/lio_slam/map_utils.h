#pragma once

#include "ad/pointcloud/point_types.h"
#include <string>

namespace sad {

/// The path to the LIO mapping results
extern std::string resultsLioMappingPath;

/// The path to the LIO mapping data (where the map tiles are stored)
extern std::string resultsLioMappingDataPath;

// Merge the LIO map into a single PCD file
// The keyframes are loaded from the map_path/keyframes.txt file.
// The keyframes are merged into a single point cloud.
// The point cloud is saved as a PCD file in the output_path directory.
int dumpLioMap(const std::string &map_path, const std::string &output_path,
               const float voxel_size, const std::string &pose_source,
               CloudPtr &global_cloud);

// Split the LIO map into smaller grids for better visualization
// The map is split into a grid of cells, each cell is grid_size meters wide.
// The map is saved as a set of PCD files, one for each cell.
// The index of the cells is saved in a text file.
// The index is the row and column of the cell in the grid.
// The PCD files are saved in the output_path directory.
// The index file is saved in the output_path directory.
int splitLioMap(const std::string &map_path, const std::string &output_path,
                const float voxel_size, const float grid_size);

// Load split map tiles (output of splitLioMap), assign a distinct color per
// grid cell, and merge into a single RGBA point cloud.
int dumpLioSplitMap(const std::string &map_data_path,
                    const std::string &output_path, UiCloudPtr &global_cloud);

} // namespace sad