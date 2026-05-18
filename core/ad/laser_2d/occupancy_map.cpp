#include "ad/laser_2d/occupancy_map.h"
#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"

#include <execution>
#include <glog/logging.h>

namespace sad {

OccupancyMap::OccupancyMap() {
  BuildModel();
  occupancy_grid_ = cv::Mat(image_size_, image_size_, CV_8U, 127);
}

void OccupancyMap::BuildModel() {
  for (int x = -model_size_; x <= model_size_; x++) {
    for (int y = -model_size_; y <= model_size_; y++) {
      Model2DPoint pt;
      pt.dx_ = x;
      pt.dy_ = y;
      pt.range_ = sqrt(x * x + y * y) * inv_resolution_;
      pt.angle_ = std::atan2(y, x);
      pt.angle_ =
          pt.angle_ > M_PI ? pt.angle_ - 2 * M_PI : pt.angle_; // limit in 2pi
      model_.push_back(pt);
    }
  }
}

double OccupancyMap::FindRangeInAngle(double angle, Scan2d::Ptr scan) {
  math::KeepAngleInPI(angle);
  if (angle < scan->angle_min || angle > scan->angle_max) {
    return 0.0;
  }

  int angle_index = int((angle - scan->angle_min) / scan->angle_increment);
  if (angle_index < 0 || angle_index >= scan->ranges.size()) {
    return 0.0;
  }

  int angle_index_p = angle_index + 1;
  double real_angle = angle;

  // take range
  double range = 0;
  if (angle_index_p >= scan->ranges.size()) {
    range = scan->ranges[angle_index];
  } else {
    // Interpolation
    double s =
        ((angle - scan->angle_min) / scan->angle_increment) - angle_index;
    double range1 = scan->ranges[angle_index];
    double range2 = scan->ranges[angle_index_p];

    double real_angle1 = scan->angle_min + scan->angle_increment * angle_index;
    double real_angle2 =
        scan->angle_min + scan->angle_increment * angle_index_p;

    if (range2 < scan->range_min || range2 > scan->range_max) {
      range = range1;
      real_angle = real_angle1;
    } else if (range1 < scan->range_min || range1 > scan->range_max) {
      range = range2;
      real_angle = real_angle2;
    } else if (std::fabs(range1 - range2) > 0.3) {
      range = s > 0.5 ? range2 : range1;
      real_angle = s > 0.5 ? real_angle2 : real_angle1;
    } else {
      range = range1 * (1 - s) + range2 * s;
    }
  }
  return range;
}

void OccupancyMap::AddLidarFrame(std::shared_ptr<Frame> frame,
                                 GridMethod method) {
  auto &scan = frame->scan_;

  // Cannot directly use frame->pose_submap_ here, because the frame may come
  // from a previous map in which case frame->pose_submap_ hasn't been updated
  // yet and still holds the pose in the previous map
  SE2 pose_in_submap = pose_.inverse() * frame->pose_;
  float theta = pose_in_submap.so2().log();
  has_outside_pts_ = false;

  // Compute the grid cells of the endpoints first
  std::set<Vec2i, less_vec<2>> endpoints;

  for (size_t i = 0; i < scan->ranges.size(); ++i) {
    if (scan->ranges[i] < scan->range_min ||
        scan->ranges[i] > scan->range_max) {
      continue;
    }

    double real_angle = scan->angle_min + i * scan->angle_increment;
    double x = scan->ranges[i] * std::cos(real_angle);
    double y = scan->ranges[i] * std::sin(real_angle);

    endpoints.emplace(World2Image(frame->pose_ * Vec2d(x, y)));
  }

  if (method == GridMethod::MODEL_POINTS) {
    // Iterate over the template, generate free points
    std::for_each(
        std::execution::par_unseq, model_.begin(), model_.end(),
        [&](const Model2DPoint &pt) {
          Vec2i pos_in_image = World2Image(frame->pose_.translation());
          Vec2i pw = pos_in_image + Vec2i(pt.dx_, pt.dy_); // submap下

          if (pt.range_ < closest_th_) {
            // Considered free at close range
            SetPoint(pw, false);
            return;
          }

          double angle = pt.angle_ - theta; // angle in lidar frame
          double range = FindRangeInAngle(angle, scan);

          if (range < scan->range_min || range > scan->range_max) {
            /// No measurement in this direction, considered invalid
            /// But mark as free when close to the sensor
            if (pt.range_ < endpoint_close_th_) {
              SetPoint(pw, false);
            }
            return;
          }

          if (range > pt.range_ && endpoints.find(pw) == endpoints.end()) {
            /// Points on the line from vehicle to endpoint, mark as free
            SetPoint(pw, false);
          }
        });
  } else {
    Vec2i start = World2Image(frame->pose_.translation());
    std::for_each(
        std::execution::par_unseq, endpoints.begin(), endpoints.end(),
        [this, &start](const auto &pt) { BresenhamFilling(start, pt); });
  }

  /// Mark endpoints as occupied
  std::for_each(endpoints.begin(), endpoints.end(),
                [this](const auto &pt) { SetPoint(pt, true); });
}

void OccupancyMap::SetPoint(const Vec2i &pt, bool occupy) {
  int x = pt[0], y = pt[1];
  if (x < 0 || y < 0 || x >= occupancy_grid_.cols ||
      y >= occupancy_grid_.rows) {
    if (occupy) {
      has_outside_pts_ = true;
    }

    return;
  }

  /// Clamped to upper/lower bounds
  uchar value = occupancy_grid_.at<uchar>(y, x);
  if (occupy) {
    if (value > 117) {
      occupancy_grid_.ptr<uchar>(y)[x] -= 1;
    }
  } else {
    if (value < 137) {
      occupancy_grid_.ptr<uchar>(y)[x] += 1;
    }
  }
}

cv::Mat OccupancyMap::GetOccupancyGridBlackWhite() const {
  cv::Mat image(image_size_, image_size_, CV_8UC3);
  for (int x = 0; x < occupancy_grid_.cols; ++x) {
    for (int y = 0; y < occupancy_grid_.rows; ++y) {
      if (occupancy_grid_.at<uchar>(y, x) == 127) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(127, 127, 127);
      } else if (occupancy_grid_.at<uchar>(y, x) < 127) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
      } else if (occupancy_grid_.at<uchar>(y, x) > 127) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
      }
    }
  }

  return image;
}

void OccupancyMap::BresenhamFilling(const Vec2i &p1, const Vec2i &p2) {
  int dx = p2.x() - p1.x();
  int dy = p2.y() - p1.y();
  int ux = dx > 0 ? 1 : -1;
  int uy = dy > 0 ? 1 : -1;

  dx = abs(dx);
  dy = abs(dy);
  int x = p1.x();
  int y = p1.y();

  if (dx > dy) {
    // increment along x
    int e = -dx;
    for (int i = 0; i < dx; ++i) {
      x += ux;
      e += 2 * dy;
      if (e >= 0) {
        y += uy;
        e -= 2 * dx;
      }

      if (Vec2i(x, y) != p2) {
        SetPoint(Vec2i(x, y), false);
      }
    }
  } else {
    int e = -dy;
    for (int i = 0; i < dy; ++i) {
      y += uy;
      e += 2 * dx;
      if (e >= 0) {
        x += ux;
        e -= 2 * dy;
      }
      if (Vec2i(x, y) != p2) {
        SetPoint(Vec2i(x, y), false);
      }
    }
  }
}

} // namespace sad