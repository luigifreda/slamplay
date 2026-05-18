//
// Created by xiang on 2021/8/9.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ad/pointcloud/point_cloud_utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

DEFINE_string(pcd_path, dataDir + "/ad/pcd/map_example.pcd",
              "point cloud file path");
DEFINE_double(image_resolution, 0.1, "bird's-eye view resolution");
DEFINE_double(min_z, 0.2, "bird's-eye view minimum height");
DEFINE_double(max_z, 2.5, "bird's-eye view maximum height");

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
  cv::Mat image = sad::GenerateBEVImage<PointCloudType>(
      cloud, FLAGS_image_resolution, FLAGS_min_z, FLAGS_max_z);

  cv::imshow("bird's-eye view", image);
  cv::waitKey(0);

  return 0;
}
