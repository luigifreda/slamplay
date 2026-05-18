//
// Created by xiang on 2021/7/20.
//
#include "ad/io/io_utils.h"

#include "ad/common/global_flags.h"
#include "ad/common/math_utils.h"
#include "ad/common/message_def.h"

#include <glog/logging.h>

namespace sad {

// read the data text file and invoke set function callbacks for each data type
void TxtIO::Go() {
  if (!fin) {
    LOG(ERROR) << "Failed to find file";
    return;
  }

  while (!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    if (line.empty()) {
      continue;
    }

    if (line[0] == '#') {
      // Lines starting with # are comments
      continue;
    }

    // load data from line
    std::stringstream ss;
    ss << line;
    std::string data_type;
    ss >> data_type;

    if (data_type == "IMU" && imu_proc_) {
      double time, gx, gy, gz, ax, ay, az;
      ss >> time >> gx >> gy >> gz >> ax >> ay >> az;
      // imu_proc_(IMU(time, Vec3d(gx, gy, gz) * math::kDEG2RAD, Vec3d(ax, ay,
      // az)));
      imu_proc_(IMU(time, Vec3d(gx, gy, gz), Vec3d(ax, ay, az)));
    } else if (data_type == "ODOM" && odom_proc_) {
      double time, wl, wr;
      ss >> time >> wl >> wr;
      odom_proc_(Odom(time, wl, wr));
    } else if (data_type == "GNSS" && gnss_proc_) {
      double time, lat, lon, alt, heading;
      bool heading_valid;
      ss >> time >> lat >> lon >> alt >> heading >> heading_valid;
      gnss_proc_(GNSS(time, 4, Vec3d(lat, lon, alt), heading, heading_valid));
    }
  }

  LOG(INFO) << "done.";
}

std::string RosbagIO::GetLidarTopicName() const {
  if (dataset_type_ == DatasetType::NCLT) {
    return nclt_lidar_topic;
  }
  if (dataset_type_ == DatasetType::ULHK) {
    return ulhk_lidar_topic;
  }
  if (dataset_type_ == DatasetType::WXB_3D) {
    return wxb_lidar_topic;
  }
  if (dataset_type_ == DatasetType::UTBM) {
    return utbm_lidar_topic;
  }
  if (dataset_type_ == DatasetType::AVIA) {
    return avia_lidar_topic;
  }

  LOG(ERROR) << "cannot load lidar topic name of dataset "
             << int(dataset_type_);
  return "";
}

void RosbagIO::Go() {
  rosbag::Bag bag;
  try {
    LOG(INFO) << "opening " << bag_file_;
    bag.open(bag_file_, rosbag::bagmode::Read);
  } catch (const rosbag::BagException &e) {
    LOG(ERROR) << "cannot open " << bag_file_ << ": " << e.what();
    return;
  }

  LOG(INFO) << "running in " << bag_file_
            << ", reg process func: " << process_func_.size();

  if (!bag.isOpen()) {
    LOG(ERROR) << "cannot open " << bag_file_;
    return;
  }

  auto view = rosbag::View(bag);
  for (const rosbag::MessageInstance &m : view) {
    auto iter = process_func_.find(m.getTopic());
    if (iter != process_func_.end()) {
      iter->second(m);
    }

    if (global::FLAG_EXIT) {
      break;
    }
  }

  bag.close();
  LOG(INFO) << "bag " << bag_file_ << " finished.";
}

RosbagIO &RosbagIO::AddImuHandle(RosbagIO::ImuHandle f) {
  return AddHandle(
      GetIMUTopicName(), [&f, this](const rosbag::MessageInstance &m) -> bool {
        auto msg = m.template instantiate<sensor_msgs::Imu>();
        if (msg == nullptr) {
          return false;
        }

        IMUPtr imu;
        if (dataset_type_ == DatasetType::AVIA) {
          // The Livox built-in IMU acceleration must be scaled by gravity
          imu = std::make_shared<IMU>(
              msg->header.stamp.toSec(),
              Vec3d(msg->angular_velocity.x, msg->angular_velocity.y,
                    msg->angular_velocity.z),
              Vec3d(msg->linear_acceleration.x * 9.80665,
                    msg->linear_acceleration.y * 9.80665,
                    msg->linear_acceleration.z * 9.80665));
        } else {
          imu = std::make_shared<IMU>(
              msg->header.stamp.toSec(),
              Vec3d(msg->angular_velocity.x, msg->angular_velocity.y,
                    msg->angular_velocity.z),
              Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                    msg->linear_acceleration.z));
        }
        return f(imu);
      });
}

std::string RosbagIO::GetIMUTopicName() const {
  if (dataset_type_ == DatasetType::ULHK) {
    return ulhk_imu_topic;
  } else if (dataset_type_ == DatasetType::UTBM) {
    return utbm_imu_topic;
  } else if (dataset_type_ == DatasetType::NCLT) {
    return nclt_imu_topic;
  } else if (dataset_type_ == DatasetType::WXB_3D) {
    return wxb_imu_topic;
  } else if (dataset_type_ == DatasetType::AVIA) {
    return avia_imu_topic;
  } else {
    LOG(ERROR) << "cannot load imu topic name of dataset "
               << int(dataset_type_);
  }

  return "";
}
} // namespace sad