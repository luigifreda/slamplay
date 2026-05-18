// // Created by xiang on 2021/8/9.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include "ad/pointcloud/point_cloud_utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

DEFINE_string(pcd_path, dataDir + "/ad/pcd/scan_example.pcd",
              "point cloud file path");
DEFINE_double(azimuth_resolution_deg, 0.3, "azimuth resolution (degrees)");
DEFINE_int32(elevation_rows, 16, "number of rows for elevation angle");
DEFINE_double(elevation_range, 15.0,
              "elevation angle range"); // VLP-16: +/-15 degrees
DEFINE_double(lidar_height, 1.128, "lidar mounting height");

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
  cv::Mat image = sad::GenerateRangeImage<PointCloudType>(
      cloud, FLAGS_azimuth_resolution_deg, FLAGS_elevation_rows,
      FLAGS_elevation_range, FLAGS_lidar_height);

  cv::namedWindow("range image", cv::WINDOW_NORMAL);
  cv::imshow("range image", image);
  cv::waitKey(0);

  return 0;
}
