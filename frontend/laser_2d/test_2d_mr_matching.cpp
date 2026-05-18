//
// Created by xiang on 2022/3/15.
//
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ad/laser_2d/frame.h"
#include "ad/laser_2d/lidar_2d_utils.h"
#include "ad/laser_2d/multi_resolution_likelihood_field.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

/// Test multi-resolution matching
/// First, you need to run test_2d_mapping.cpp to generate the submap images and
/// loop constraints:
//  $ ./test_2d_mapping.cpp --with_loop_closing=true

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "First, you need to run test_2d_mapping.cpp to generate the "
               "submap images and loop constraints.\n"
               "./test_2d_mapping.cpp --with_loop_closing=true";

  std::ifstream fin(resultsDir + "/ad/2dmapping/loops.txt");
  int loop_id = 0;

  while (!fin.eof()) {
    int frame_id, submap_id;
    double submap_center_x, submap_center_y, theta;
    if (fin.peek() == fin.eof()) {
      break;
    }

    fin >> frame_id >> submap_id >> submap_center_x >> submap_center_y >> theta;
    loop_id++;

    sad::MRLikelihoodField mr_field;
    Vec2d center(submap_center_x, submap_center_y);
    SE2 pose_submap(SO2::exp(theta), center);

    mr_field.SetPose(pose_submap);
    cv::Mat occu_map = cv::imread(resultsDir + "/ad/2dmapping/submap_" +
                                      std::to_string(submap_id) + ".png",
                                  cv::IMREAD_GRAYSCALE);
    cv::Mat occu_map_color = cv::imread(resultsDir + "/ad/2dmapping/submap_" +
                                            std::to_string(submap_id) + ".png",
                                        cv::IMREAD_COLOR);
    mr_field.SetFieldImageFromOccuMap(occu_map);

    sad::Frame frame;
    frame.Load(resultsDir + "/ad/2dmapping/frame_" + std::to_string(frame_id) +
               ".txt");
    mr_field.SetSourceScan(frame.scan_);

    LOG(INFO) << "testing frame " << frame.id_ << " with " << submap_id;

    auto init_pose = frame.pose_;
    auto frame_pose_in_submap = pose_submap.inverse() * frame.pose_;
    bool align_success = mr_field.AlignG2O(frame_pose_in_submap);

    if (align_success) {
      frame.pose_ = pose_submap * frame_pose_in_submap;
      auto images = mr_field.GetFieldImage();
      for (int i = 0; i < images.size(); ++i) {
        /// Initial pose shown in red
        sad::Visualize2DScan(frame.scan_, init_pose, images[i],
                             Vec3b(0, 0, 255), images[i].rows,
                             mr_field.Resolution(i), pose_submap);
        /// Aligned pose shown in green
        sad::Visualize2DScan(frame.scan_, frame.pose_, images[i],
                             Vec3b(0, 255, 0), images[i].rows,
                             mr_field.Resolution(i), pose_submap);
        cv::imshow("level " + std::to_string(i), images[i]);
      }

      sad::Visualize2DScan(frame.scan_, init_pose, occu_map_color,
                           Vec3b(0, 0, 255), occu_map_color.rows, 20.0,
                           pose_submap);
      sad::Visualize2DScan(frame.scan_, frame.pose_, occu_map_color,
                           Vec3b(0, 255, 0), occu_map_color.rows, 20.0,
                           pose_submap);
      cv::imshow("occupancy", occu_map_color);
      LOG(INFO) << "frame " << frame.id_ << " aligned with submap "
                << submap_id;
      LOG(INFO) << "press any key to continue";
      cv::waitKey();
    }
  }

  return 0;
}