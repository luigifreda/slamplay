#include "ad/laser_2d/mapping_2d.h"
#include "ad/laser_2d/lidar_2d_utils.h"
#include "ad/laser_2d/loop_closing.h"
#include "ad/laser_2d/submap.h"

#include <execution>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include "macros.h"

namespace {
std::string mappingResultsDir =
    STR(RESULTS_DIR) +
    std::string("/ad/2dmapping"); // RESULTS_DIR set by compilers flag
}

namespace sad {

bool Mapping2D::Init(bool with_loop_closing) {
  keyframe_id_ = 0;
  current_submap_ = std::make_shared<Submap>(SE2());
  all_submaps_.emplace_back(current_submap_);

  if (with_loop_closing) {
    loop_closing_ = std::make_shared<LoopClosing>();
    loop_closing_->AddNewSubmap(current_submap_);
  }

  std::string resultsPathDir = mappingResultsDir;
  if (!std::filesystem::path(resultsPathDir).empty()) {
    std::filesystem::create_directories(resultsPathDir);
  }

  return true;
}

bool Mapping2D::ProcessScan(MultiScan2d::Ptr scan) {
  return ProcessScan(MultiToScan2d(scan));
}

bool Mapping2D::ProcessScan(Scan2d::Ptr scan) {
  current_frame_ = std::make_shared<Frame>(scan);
  current_frame_->id_ = frame_id_++;

  if (last_frame_) {
    // set pose from last frame
    // current_frame_->pose_ = last_frame_->pose_;
    current_frame_->pose_ = last_frame_->pose_ * motion_guess_;
    current_frame_->pose_submap_ = last_frame_->pose_submap_;
  }

  // Use scan matching to align with the map
  if (!first_scan_) {
    // First frame cannot be matched, add directly to the occupancy map
    current_submap_->MatchScan(current_frame_);
  }

  // current_submap_->AddScanInOccupancyMap(current_frame_);
  first_scan_ = false;
  bool is_kf = IsKeyFrame();

  if (is_kf) {
    AddKeyFrame();
    current_submap_->AddScanInOccupancyMap(current_frame_);

    // Handle loop closing
    if (loop_closing_) {
      loop_closing_->AddNewFrame(current_frame_);
    }

    if (current_submap_->HasOutsidePoints() ||
        (current_submap_->NumFrames()) > 50) {
      /// Exited the submap or too many keyframes in a single submap
      ExpandSubmap();
    }
  }

  /// Visualization output
  auto occu_image = current_submap_->GetOccuMap().GetOccupancyGridBlackWhite();
  Visualize2DScan(current_frame_->scan_, current_frame_->pose_, occu_image,
                  Vec3b(0, 0, 255), 1000, 20.0, current_submap_->GetPose());
  cv::putText(occu_image, "submap " + std::to_string(current_submap_->GetId()),
              cv::Point2f(20, 20), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(0, 255, 0));
  cv::putText(occu_image,
              "keyframes " + std::to_string(current_submap_->NumFrames()),
              cv::Point2f(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(0, 255, 0));
  cv::imshow("occupancy map", occu_image);

  auto field_image = current_submap_->GetLikelihood().GetFieldImage();
  Visualize2DScan(current_frame_->scan_, current_frame_->pose_, field_image,
                  Vec3b(0, 0, 255), 1000, 20.0, current_submap_->GetPose());
  cv::imshow("likelihood", field_image);

  /// global map
  if (is_kf) {
    cv::imshow("global map", ShowGlobalMap());
  }

  cv::waitKey(10);

  if (last_frame_) {
    motion_guess_ = last_frame_->pose_.inverse() * current_frame_->pose_;
  }

  last_frame_ = current_frame_;

  return true;
}

bool Mapping2D::IsKeyFrame() {
  if (last_keyframe_ == nullptr) {
    return true;
  }

  SE2 delta_pose = last_keyframe_->pose_.inverse() * current_frame_->pose_;
  if (delta_pose.translation().norm() > keyframe_pos_th_ ||
      fabs(delta_pose.so2().log()) > keyframe_ang_th_) {
    return true;
  }

  return false;
}

void Mapping2D::AddKeyFrame() {
  LOG(INFO) << "add keyframe " << keyframe_id_;
  current_frame_->keyframe_id_ = keyframe_id_++;
  current_submap_->AddKeyFrame(current_frame_);
  last_keyframe_ = current_frame_;
}

void Mapping2D::ExpandSubmap() {
  // Pass the current submap to loop closing as a historical map
  if (loop_closing_) {
    loop_closing_->AddFinishedSubmap(current_submap_);
  }

  // Replace the current submap with a new one
  auto last_submap = current_submap_;

  // debug
  cv::imwrite(mappingResultsDir + "/submap_" +
                  std::to_string(last_submap->GetId()) + ".png",
              last_submap->GetOccuMap().GetOccupancyGridBlackWhite());

  current_submap_ = std::make_shared<Submap>(current_frame_->pose_);
  current_frame_->pose_submap_ = SE2(); // reset to identity

  current_submap_->SetId(++submap_id_);
  current_submap_->AddKeyFrame(current_frame_);
  current_submap_->SetOccuFromOtherSubmap(
      last_submap); // copy data from previous submap so the new one doesn't
                    // appear empty

  current_submap_->AddScanInOccupancyMap(current_frame_);
  all_submaps_.emplace_back(current_submap_);

  if (loop_closing_) {
    loop_closing_->AddNewSubmap(current_submap_);
  }

  LOG(INFO) << "create submap " << current_submap_->GetId() << " with pose: "
            << current_submap_->GetPose().translation().transpose() << ", "
            << current_submap_->GetPose().so2().log();
}

cv::Mat Mapping2D::ShowGlobalMap(int max_size) {
  //// TODO: fixed-size global map with dynamic resolution
  Vec2f top_left = Vec2f(999999, 999999);
  Vec2f bottom_right = Vec2f(-999999, -999999);

  const float submap_resolution = 20.0; // submap resolution (pixels per meter)
  const float submap_size = 50.0;       // single submap size

  /// Compute physical bounds of the global map
  for (auto m : all_submaps_) {
    Vec2d c = m->GetPose().translation();
    if (top_left[0] > c[0] - submap_size / 2) {
      top_left[0] = c[0] - submap_size / 2;
    }
    if (top_left[1] > c[1] - submap_size / 2) {
      top_left[1] = c[1] - submap_size / 2;
    }

    if (bottom_right[0] < c[0] + submap_size / 2) {
      bottom_right[0] = c[0] + submap_size / 2;
    }
    if (bottom_right[1] < c[1] + submap_size / 2) {
      bottom_right[1] = c[1] + submap_size / 2;
    }
  }

  if (top_left[0] > bottom_right[0] || top_left[1] > bottom_right[1]) {
    return cv::Mat();
  }

  /// Physical center of the global map
  Vec2f global_center = Vec2f((top_left[0] + bottom_right[0]) / 2.0,
                              (top_left[1] + bottom_right[1]) / 2.0);
  float phy_width = bottom_right[0] - top_left[0];  // physical width
  float phy_height = bottom_right[1] - top_left[1]; // physical height
  float global_map_resolution = 0;

  if (phy_width > phy_height) {
    global_map_resolution = max_size / phy_width;
  } else {
    global_map_resolution = max_size / phy_height;
  }

  Vec2f c = global_center;
  int c_x = global_center[0] * global_map_resolution;
  int c_y = global_center[1] * global_map_resolution;
  global_center = Vec2f(c_x / global_map_resolution,
                        c_y / global_map_resolution); // global map image center

  int width =
      int((bottom_right[0] - top_left[0]) * global_map_resolution + 0.5);
  int height =
      int((bottom_right[1] - top_left[1]) * global_map_resolution + 0.5);

  Vec2f center_image = Vec2f(width / 2, height / 2);
  cv::Mat output_image(height, width, CV_8UC3, cv::Scalar(127, 127, 127));

  std::vector<Vec2i> render_data;
  render_data.reserve(width * height);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      render_data.emplace_back(Vec2i(x, y));
    }
  }

  std::for_each(
      std::execution::par_unseq, render_data.begin(), render_data.end(),
      [&](const Vec2i &xy) {
        int x = xy[0], y = xy[1];
        Vec2f pw = (Vec2f(x, y) - center_image) / global_map_resolution +
                   c; // world coordinates

        for (auto &m : all_submaps_) {
          Vec2f ps = m->GetPose().inverse().cast<float>() * pw; // in submap
          Vec2i pt = (ps * submap_resolution + Vec2f(500, 500)).cast<int>();

          if (pt[0] < 0 || pt[0] >= 1000 || pt[1] < 0 || pt[1] >= 1000) {
            continue;
          }

          uchar value =
              m->GetOccuMap().GetOccupancyGrid().at<uchar>(pt[1], pt[0]);
          if (value > 127) {
            if (m == current_submap_) {
              output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(235, 250, 230);
            } else {
              output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
            break;
          } else if (value < 127) {
            if (m == current_submap_) {
              output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(230, 20, 30);
            } else {
              output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
            break;
          }
        }
      });

  for (auto &m : all_submaps_) {
    /// Projection of submap pose in the global map
    SE2f submap_pose = m->GetPose().cast<float>();
    Vec2f submap_center = submap_pose.translation();
    Vec2f submap_xw = submap_pose * Vec2f(1.0, 0);
    Vec2f submap_yw = submap_pose * Vec2f(0, 1.0);

    Vec2f center_map =
        (submap_center - global_center) * global_map_resolution + center_image;
    Vec2f x_map =
        (submap_xw - global_center) * global_map_resolution + center_image;
    Vec2f y_map =
        (submap_yw - global_center) * global_map_resolution + center_image;

    // x-axis and y-axis
    cv::line(output_image, cv::Point2f(center_map.x(), center_map.y()),
             cv::Point2f(x_map.x(), x_map.y()), cv::Scalar(0, 0, 255), 2);
    cv::line(output_image, cv::Point2f(center_map.x(), center_map.y()),
             cv::Point2f(y_map.x(), y_map.y()), cv::Scalar(0, 255, 0), 2);
    cv::putText(output_image, std::to_string(m->GetId()),
                cv::Point2f(center_map.x() + 10, center_map.y() - 10),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0));

    // Trajectory
    for (const auto &frame : m->GetFrames()) {
      Vec2f p_map = (frame->pose_.translation().cast<float>() - global_center) *
                        global_map_resolution +
                    center_image;
      cv::circle(output_image, cv::Point2f(p_map.x(), p_map.y()), 1,
                 cv::Scalar(0, 0, 255), 1);
    }
  }

  if (loop_closing_) {
    /// Loop closure pose graph
    auto loops = loop_closing_->GetLoops();
    for (auto lc : loops) {
      auto first_id = lc.first.first;
      auto second_id = lc.first.second;

      Vec2f c1 = all_submaps_[first_id]->GetPose().translation().cast<float>();
      Vec2f c2 = all_submaps_[second_id]->GetPose().translation().cast<float>();

      Vec2f c1_map =
          (c1 - global_center) * global_map_resolution + center_image;
      Vec2f c2_map =
          (c2 - global_center) * global_map_resolution + center_image;

      cv::line(output_image, cv::Point2f(c1_map.x(), c1_map.y()),
               cv::Point2f(c2_map.x(), c2_map.y()), cv::Scalar(255, 0, 0), 2);
    }
  }

  return output_image;
}

} // namespace sad