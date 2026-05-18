#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <string>

#include "ad/pointcloud/point_cloud_utils.h"
#include "lio_slam/keyframe.h"

#include "macros.h"

namespace {
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
} // namespace

namespace sad {
namespace {

// Convert HSV to RGB
void hsvToRgb(float h, float s, float v, uint8_t &r, uint8_t &g, uint8_t &b) {
  const int sector = static_cast<int>(h * 6.0f);
  const float f = h * 6.0f - sector;
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - f * s);
  const float t = v * (1.0f - (1.0f - f) * s);

  float rf = 0.0f;
  float gf = 0.0f;
  float bf = 0.0f;
  switch (sector % 6) {
  case 0:
    rf = v;
    gf = t;
    bf = p;
    break;
  case 1:
    rf = q;
    gf = v;
    bf = p;
    break;
  case 2:
    rf = p;
    gf = v;
    bf = t;
    break;
  case 3:
    rf = p;
    gf = q;
    bf = v;
    break;
  case 4:
    rf = t;
    gf = p;
    bf = v;
    break;
  default:
    rf = v;
    gf = p;
    bf = q;
    break;
  }

  r = static_cast<uint8_t>(rf * 255.0f);
  g = static_cast<uint8_t>(gf * 255.0f);
  b = static_cast<uint8_t>(bf * 255.0f);
}

// Assign a distinct color per grid cell. Edge-adjacent cells (|dx|+|dy|==1) use
// different hues via a 2x2 checkerboard; saturation/value vary on 2x2
// super-cells so same-parity tiles farther apart still look different.
void gridCellColor(int gx, int gy, uint8_t &r, uint8_t &g, uint8_t &b) {
  const int parity = ((gx & 1) << 1) | (gy & 1);
  static constexpr float kHue[4] = {0.0f, 0.33f, 0.58f, 0.83f};

  const int super_x = gx >> 1;
  const int super_y = gy >> 1;
  const int stripe = (super_x * 5 + super_y * 11) & 3;

  const float h = kHue[parity];
  const float s = 0.65f + 0.10f * static_cast<float>(stripe);
  const float v =
      0.85f + 0.10f * static_cast<float>((stripe + parity) & 3) / 3.0f;

  hsvToRgb(h, s, v, r, g, b);
}

// Append a colored tile to the global cloud
void appendColoredTile(const CloudPtr &tile, uint8_t r, uint8_t g, uint8_t b,
                       UiCloudPtr &global_cloud) {
  global_cloud->points.reserve(global_cloud->size() + tile->size());
  for (const auto &pt : tile->points) {
    UiPointType colored;
    colored.x = pt.x;
    colored.y = pt.y;
    colored.z = pt.z;
    colored.r = r;
    colored.g = g;
    colored.b = b;
    colored.a = 255;
    global_cloud->points.emplace_back(colored);
  }
}

} // namespace

std::string resultsLioMappingPath = resultsDir + "/ad/lio_slam/map";
std::string resultsLioMappingDataPath = resultsDir + "/ad/lio_slam/map_data";

int dumpLioMap(const std::string &map_path, const std::string &output_path,
               const float voxel_size, const std::string &pose_source,
               CloudPtr &global_cloud) {

  if (global_cloud != nullptr) {
    global_cloud->clear();
  } else {
    global_cloud.reset(new PointCloudType);
  }

  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directories(output_path);
  }

  using namespace sad;
  std::map<IdType, KFPtr> keyframes;
  if (!LoadKeyFrames(map_path + "/keyframes.txt", keyframes)) {
    LOG(ERROR) << "failed to load keyframes.txt";
    return -1;
  }

  if (keyframes.empty()) {
    LOG(INFO) << "keyframes are empty";
    return 0;
  }

  // dump kf cloud and merge
  LOG(INFO) << "merging";

  pcl::VoxelGrid<PointType> voxel_grid_filter;
  float resolution = voxel_size;
  voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

  int cnt = 0;
  for (auto &kfp : keyframes) {
    auto kf = kfp.second;
    SE3 pose;
    if (pose_source == "rtk") {
      pose = kf->rtk_pose_;
    } else if (pose_source == "lidar") {
      pose = kf->lidar_pose_;
    } else if (pose_source == "opti1") {
      pose = kf->opti_pose_1_;
    } else if (pose_source == "opti2") {
      pose = kf->opti_pose_2_;
    }

    kf->LoadScan(map_path + "/");

    CloudPtr cloud_trans(new PointCloudType);
    pcl::transformPointCloud(*kf->cloud_, *cloud_trans, pose.matrix());

    // voxel size
    CloudPtr kf_cloud_voxeled(new PointCloudType);
    voxel_grid_filter.setInputCloud(cloud_trans);
    voxel_grid_filter.filter(*kf_cloud_voxeled);

    *global_cloud += *kf_cloud_voxeled;
    kf->cloud_ = nullptr;

    LOG(INFO) << "merging " << cnt << " in " << keyframes.size()
              << ", pts: " << kf_cloud_voxeled->size()
              << " global pts: " << global_cloud->size();
    cnt++;
  }

  if (!global_cloud->empty()) {
    sad::SaveCloudToFile(output_path + "/map.pcd", *global_cloud);
  }

  LOG(INFO) << "done.";
  return 0;
}

int splitLioMap(const std::string &map_path, const std::string &output_path,
                const float voxel_size, const float grid_size) {

  using namespace sad;

  std::map<IdType, KFPtr> keyframes;
  if (!LoadKeyFrames(map_path + "/keyframes.txt", keyframes)) {
    LOG(ERROR) << "failed to load keyframes";
    return 0;
  }

  std::map<Vec2i, CloudPtr, less_vec<2>>
      map_data; // map data indexed by grid ID
  pcl::VoxelGrid<PointType> voxel_grid_filter;
  float resolution = voxel_size;
  voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

  // Each point looks up its grid ID; create one if missing
  for (auto &kfp : keyframes) {
    auto kf = kfp.second;
    kf->LoadScan(map_path + "/");

    CloudPtr cloud_trans(new PointCloudType);
    pcl::transformPointCloud(*kf->cloud_, *cloud_trans,
                             kf->opti_pose_2_.matrix());

    // voxel size
    CloudPtr kf_cloud_voxeled(new PointCloudType);
    voxel_grid_filter.setInputCloud(cloud_trans);
    voxel_grid_filter.filter(*kf_cloud_voxeled);

    LOG(INFO) << "building kf " << kf->id_ << " in " << keyframes.size();

    const float grid_size_half = grid_size / 2;

    // add to grid
    for (const auto &pt : kf_cloud_voxeled->points) {
      //   int gx = floor((pt.x - 50.0) / 100);
      //   int gy = floor((pt.y - 50.0) / 100);
      int gx = int(floor((pt.x - grid_size_half) / grid_size));
      int gy = int(floor((pt.y - grid_size_half) / grid_size));
      Vec2i key(gx, gy);
      auto iter = map_data.find(key);
      if (iter == map_data.end()) {
        // create point cloud
        CloudPtr cloud(new PointCloudType);
        cloud->points.emplace_back(pt);
        cloud->is_dense = false;
        cloud->height = 1;
        map_data.emplace(key, cloud);
      } else {
        iter->second->points.emplace_back(pt);
      }
    }
  }

  // Save point clouds and index file
  LOG(INFO) << "saving maps, grids: " << map_data.size();
  std::system(("mkdir -p " + output_path + "/").c_str());
  std::system(("rm -rf " + output_path + "/*").c_str()); // clean output folder
  std::ofstream fout(output_path + "/map_index.txt");
  for (auto &dp : map_data) {
    fout << dp.first[0] << " " << dp.first[1] << std::endl;
    dp.second->width = dp.second->size();
    sad::VoxelGrid(dp.second, 0.1);

    sad::SaveCloudToFile(output_path + "/" + std::to_string(dp.first[0]) + "_" +
                             std::to_string(dp.first[1]) + ".pcd",
                         *dp.second);
  }
  fout.close();

  return 0;
}

int dumpLioSplitMap(const std::string &map_data_path,
                    const std::string &output_path, UiCloudPtr &global_cloud) {
  if (global_cloud != nullptr) {
    global_cloud->clear();
  } else {
    global_cloud.reset(new UiPointCloudType);
  }

  const std::string index_path = map_data_path + "/map_index.txt";
  if (!std::filesystem::exists(index_path)) {
    LOG(ERROR) << "missing map index: " << index_path;
    return -1;
  }

  std::ifstream fin(index_path);
  int gx = 0;
  int gy = 0;
  int tile_count = 0;
  while (fin >> gx >> gy) {
    const std::string tile_path = map_data_path + "/" + std::to_string(gx) +
                                  "_" + std::to_string(gy) + ".pcd";
    CloudPtr tile(new PointCloudType);
    if (pcl::io::loadPCDFile(tile_path, *tile) < 0) {
      LOG(ERROR) << "failed to load tile: " << tile_path;
      return -1;
    }

    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    gridCellColor(gx, gy, r, g, b);
    appendColoredTile(tile, r, g, b, global_cloud);
    tile_count++;

    LOG(INFO) << "loaded tile (" << gx << ", " << gy
              << "), pts: " << tile->size()
              << ", global pts: " << global_cloud->size();
  }

  if (tile_count == 0) {
    LOG(INFO) << "no map tiles found in " << index_path;
    return 0;
  }

  global_cloud->width = global_cloud->size();
  global_cloud->height = 1;
  global_cloud->is_dense = false;

  if (!output_path.empty()) {
    if (!std::filesystem::exists(output_path)) {
      std::filesystem::create_directories(output_path);
    }
    pcl::io::savePCDFileASCII(output_path + "/map_colored.pcd", *global_cloud);
  }

  LOG(INFO) << "done, tiles: " << tile_count
            << ", pts: " << global_cloud->size();
  return 0;
}

} // namespace sad