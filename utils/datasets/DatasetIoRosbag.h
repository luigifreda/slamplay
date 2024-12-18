/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <mutex>
#include <optional>

#include "datasets/DatasetIo.h"

// Hack to access private functions
#define private public
#include <rosbag/bag.h>
#include <rosbag/view.h>
#undef private

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include "io/filesystem.h"
#include "io/messages.h"

namespace slamplay {

class RosbagVioDataset : public VioDataset {
    std::shared_ptr<rosbag::Bag> bag;
    std::mutex m;

    size_t num_cams = 0;
    size_t num_depth_cams = 0;

    std::vector<int64_t> image_timestamps;
    std::vector<int64_t> depth_timestamps;

    // vector of images for every timestamp
    // assumes vectors size is num_cams for every timestamp with null pointers for
    // missing frames
    std::unordered_map<int64_t, std::vector<std::optional<rosbag::IndexEntry>>> image_data_idx;

    std::unordered_map<int64_t, std::vector<std::optional<rosbag::IndexEntry>>> depth_data_idx;

    Eigen::aligned_vector<AccelData> accel_data;
    Eigen::aligned_vector<GyroData> gyro_data;

    std::vector<int64_t> gt_timestamps;                // ordered gt timestamps
    Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;  // TODO: change to eigen aligned

    int64_t mocap_to_imu_offset_ns;

   public:
    ~RosbagVioDataset() {}

    size_t get_num_cams() const { return num_cams; }
    size_t get_num_depth_cams() const { return num_depth_cams; }

    std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }
    std::vector<int64_t> &get_depth_timestamps() { return depth_timestamps; }

    const Eigen::aligned_vector<AccelData> &get_accel_data() const {
        return accel_data;
    }
    const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
        return gyro_data;
    }
    const std::vector<int64_t> &get_gt_timestamps() const {
        return gt_timestamps;
    }
    const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
        return gt_pose_data;
    }

    int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

    std::vector<ImageData> get_image_data(int64_t t_ns) {
        std::vector<ImageData> res(num_cams);

        auto it = image_data_idx.find(t_ns);

        if (it != image_data_idx.end())
        {
            for (size_t i = 0; i < num_cams; i++) {
                ImageData &id = res[i];

                if (!it->second[i].has_value()) continue;

                m.lock();
                sensor_msgs::ImageConstPtr img_msg = bag->instantiateBuffer<sensor_msgs::Image>(*it->second[i]);
                m.unlock();

                if (!img_msg) {
                    std::cerr << "Error instantiating image message on timestamp " << t_ns << std::endl;
                    continue;  // Skip this message if instantiation fails
                }

                if (!img_msg->header.frame_id.empty() &&
                    std::isdigit(img_msg->header.frame_id[0])) {
                    id.exposure = std::stol(img_msg->header.frame_id) * 1e-9;
                } else {
                    id.exposure = -1;
                }

                if (img_msg->encoding == "mono8") {
                    id.img = cv::Mat(img_msg->height, img_msg->width, CV_8UC1);
                    MSG_ASSERT(img_msg->data.size() == id.img.step[0] * id.img.rows, "Image size mismatch");
                    std::memcpy(id.img.data, img_msg->data.data(), img_msg->data.size());
                } else if (img_msg->encoding == "mono16") {
                    id.img = cv::Mat(img_msg->height, img_msg->width, CV_16UC1);
                    MSG_ASSERT(img_msg->data.size() == id.img.step[0] * id.img.rows, "Image size mismatch");
                    std::memcpy(id.img.data, img_msg->data.data(), img_msg->data.size());
                } else if (img_msg->encoding == "rgb8") {
                    id.img = cv::Mat(img_msg->height, img_msg->width, CV_8UC3);
                    MSG_ASSERT(img_msg->data.size() == id.img.step[0] * id.img.rows, "Image size mismatch");
                    std::memcpy(id.img.data, img_msg->data.data(), img_msg->data.size());
                } else if (img_msg->encoding == "bgra8") {
                    id.img = cv::Mat(img_msg->height, img_msg->width, CV_8UC4);
                    MSG_ASSERT(img_msg->data.size() == id.img.step[0] * id.img.rows, "Image size mismatch");
                    std::memcpy(id.img.data, img_msg->data.data(), img_msg->data.size());
                } else {
                    std::cerr << "Encoding " << img_msg->encoding << " is not supported." << std::endl;
                    std::abort();
                }
            }
        }

        return res;
    }

    std::vector<DepthData> get_depth_data(int64_t t_ns) {
        std::vector<DepthData> res(num_depth_cams);

        auto it = depth_data_idx.find(t_ns);

        if (it != depth_data_idx.end())
        {
            for (size_t i = 0; i < num_depth_cams; i++) {
                DepthData &dd = res[i];

                if (!it->second[i].has_value()) continue;

                m.lock();
                sensor_msgs::ImageConstPtr depth_msg = bag->instantiateBuffer<sensor_msgs::Image>(*it->second[i]);
                m.unlock();

                if (!depth_msg) {
                    std::cerr << "Error instantiating depth message on timestamp " << t_ns << std::endl;
                    continue;  // Skip this message if instantiation fails
                }

                if (depth_msg->encoding == "mono16") {
                    dd.img = cv::Mat(depth_msg->height, depth_msg->width, CV_16UC1);
                    MSG_ASSERT(depth_msg->data.size() == dd.img.step[0] * dd.img.rows, "Depth Image size mismatch");
                    std::memcpy(dd.img.data, depth_msg->data.data(), depth_msg->data.size());
                } else if (depth_msg->encoding == "32FC1") {
                    dd.img = cv::Mat(depth_msg->height, depth_msg->width, CV_32FC1);
                    MSG_ASSERT(depth_msg->data.size() == dd.img.step[0] * dd.img.rows, "Depth Image size mismatch");
                    std::memcpy(dd.img.data, depth_msg->data.data(), depth_msg->data.size());
                } else {
                    std::cerr << "Unsupported depth encoding: " << depth_msg->encoding << std::endl;
                    std::abort();
                }
            }
        }

        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class RosbagIO;
};

class RosbagIO : public DatasetIoInterface {
   public:
    RosbagIO() {}

    void read(const std::string &path) {
        if (!fs::exists(path)) {
            std::cerr << "No dataset found in " << path << std::endl;
            return;
        }

        data.reset(new RosbagVioDataset);

        data->bag.reset(new rosbag::Bag);
        try {
            data->bag->open(path, rosbag::bagmode::Read);
        } catch (const rosbag::BagException &e) {
            std::cerr << "Error opening bag file: " << e.what() << std::endl;
            return;  // Handle bag opening error
        }

        rosbag::View view(*data->bag);

        // get topics
        std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();

        std::set<std::string> cam_topics;
        std::set<std::string> depth_topics;
        std::string imu_topic;
        std::string mocap_topic;
        std::string point_topic;

        for (const rosbag::ConnectionInfo *info : connection_infos) {
            if (info->datatype == std::string("sensor_msgs/Image")) {
                if (info->topic.find("depth") != std::string::npos) {
                    depth_topics.insert(info->topic);
                } else {
                    cam_topics.insert(info->topic);
                }
            } else if (info->datatype == std::string("sensor_msgs/Imu") &&
                       info->topic.rfind("/fcu", 0) != 0) {
                imu_topic = info->topic;
            } else if (info->datatype ==
                           std::string("geometry_msgs/TransformStamped") ||
                       info->datatype == std::string("geometry_msgs/PoseStamped")) {
                mocap_topic = info->topic;
            } else if (info->datatype == std::string("geometry_msgs/PointStamped")) {
                point_topic = info->topic;
            }
        }

        std::cout << "imu_topic: " << imu_topic << std::endl;
        std::cout << "mocap_topic: " << mocap_topic << std::endl;
        std::cout << "cam_topics: ";
        for (const std::string &s : cam_topics) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "depth_topics: ";
        for (const std::string &s : depth_topics) std::cout << s << " ";
        std::cout << std::endl;

        std::map<std::string, int> image_topic_to_id;
        int idx = 0;
        for (const std::string &s : cam_topics) {
            image_topic_to_id[s] = idx;
            std::cout << "image_topic_to_id[" << s << "] = " << idx << std::endl;
            idx++;
        }
        idx = 0;
        std::map<std::string, int> depth_topic_to_id;
        for (const std::string &s : depth_topics) {
            depth_topic_to_id[s] = idx;
            std::cout << "depth_topic_to_id[" << s << "] = " << idx << std::endl;
            idx++;
        }

        data->num_cams = cam_topics.size();
        data->num_depth_cams = depth_topics.size();
        std::cout << "num_cams: " << data->num_cams << std::endl;
        std::cout << "num_depth_cams: " << data->num_depth_cams << std::endl;

        int num_msgs = 0;

        int64_t min_time = std::numeric_limits<int64_t>::max();
        int64_t max_time = std::numeric_limits<int64_t>::min();

        std::vector<geometry_msgs::TransformStampedConstPtr> mocap_msgs;
        std::vector<geometry_msgs::PointStampedConstPtr> point_msgs;

        std::vector<int64_t> system_to_imu_offset_vec;    // t_imu = t_system + system_to_imu_offset
        std::vector<int64_t> system_to_mocap_offset_vec;  // t_mocap = t_system +
                                                          // system_to_mocap_offset

        std::set<int64_t> image_timestamps;
        std::set<int64_t> depth_timestamps;

#define VERBOSE 0

        for (const rosbag::MessageInstance &m : view) {
            const std::string &topic = m.getTopic();

#if VERBOSE
            std::cout << "msg " << num_msgs << ": " << topic << std::endl;
#endif

            if (cam_topics.find(topic) != cam_topics.end()) {
#if VERBOSE
                std::cout << "getting cam image..." << std::endl;
#endif
                sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                if (!img_msg) {
                    std::cerr << "Error instantiating image message on topic " << topic << std::endl;
                    continue;  // Skip this message if instantiation fails
                }
                int64_t timestamp_ns = img_msg->header.stamp.toNSec();

                auto &img_vec = data->image_data_idx[timestamp_ns];
                if (img_vec.empty()) {
                    img_vec.resize(data->num_cams);
                }

                img_vec[image_topic_to_id.at(topic)] = m.index_entry_;
                image_timestamps.insert(timestamp_ns);

                min_time = std::min(min_time, timestamp_ns);
                max_time = std::max(max_time, timestamp_ns);
#if VERBOSE
                std::cout << "...done" << std::endl;
#endif
            }

            if (depth_topics.find(topic) != depth_topics.end()) {
#if VERBOSE
                std::cout << "getting depth image..." << std::endl;
#endif
                sensor_msgs::ImageConstPtr depth_msg = m.instantiate<sensor_msgs::Image>();
                if (!depth_msg) {
                    std::cerr << "Error instantiating depth message on topic " << topic << std::endl;
                    continue;  // Skip this message if instantiation fails
                }
                int64_t timestamp_ns = depth_msg->header.stamp.toNSec();

                auto &depth_vec = data->depth_data_idx[timestamp_ns];
                if (depth_vec.empty()) {
                    depth_vec.resize(data->num_depth_cams);
                }

                depth_vec[depth_topic_to_id.at(topic)] = m.index_entry_;
                depth_timestamps.insert(timestamp_ns);

                min_time = std::min(min_time, timestamp_ns);
                max_time = std::max(max_time, timestamp_ns);
#if VERBOSE
                std::cout << "...done" << std::endl;
#endif
            }

            if (imu_topic == topic) {
#if VERBOSE
                std::cout << "getting imu data ..." << std::endl;
#endif
                sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                if (!imu_msg) {
                    std::cerr << "Error instantiating IMU message" << std::endl;
                    continue;  // Skip this message if instantiation fails
                }
                int64_t time = imu_msg->header.stamp.toNSec();

                data->accel_data.emplace_back();
                data->accel_data.back().timestamp_ns = time;
                data->accel_data.back().data = Eigen::Vector3d(
                    imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
                    imu_msg->linear_acceleration.z);

                data->gyro_data.emplace_back();
                data->gyro_data.back().timestamp_ns = time;
                data->gyro_data.back().data = Eigen::Vector3d(
                    imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                    imu_msg->angular_velocity.z);

                min_time = std::min(min_time, time);
                max_time = std::max(max_time, time);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_imu_offset_vec.push_back(time - msg_arrival_time);
#if VERBOSE
                std::cout << "...done" << std::endl;
#endif
            }

            if (mocap_topic == topic) {
#if VERBOSE
                std::cout << "getting mocap data ..." << std::endl;
#endif
                geometry_msgs::TransformStampedConstPtr mocap_msg =
                    m.instantiate<geometry_msgs::TransformStamped>();

                // Try different message type if instantiate did not work
                if (!mocap_msg) {
                    geometry_msgs::PoseStampedConstPtr mocap_pose_msg =
                        m.instantiate<geometry_msgs::PoseStamped>();
                    if (!mocap_pose_msg) {
                        std::cerr << "Error instantiating mocap message" << std::endl;
                        continue;  // Skip this message if instantiation fails
                    }

                    geometry_msgs::TransformStampedPtr mocap_new_msg(
                        new geometry_msgs::TransformStamped);
                    mocap_new_msg->header = mocap_pose_msg->header;
                    mocap_new_msg->transform.rotation = mocap_pose_msg->pose.orientation;
                    mocap_new_msg->transform.translation.x =
                        mocap_pose_msg->pose.position.x;
                    mocap_new_msg->transform.translation.y =
                        mocap_pose_msg->pose.position.y;
                    mocap_new_msg->transform.translation.z =
                        mocap_pose_msg->pose.position.z;

                    mocap_msg = mocap_new_msg;
                }

                int64_t time = mocap_msg->header.stamp.toNSec();

                mocap_msgs.push_back(mocap_msg);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
#if VERBOSE
                std::cout << "...done" << std::endl;
#endif
            }

            if (point_topic == topic) {
#if VERBOSE
                std::cout << "getting point data ..." << std::endl;
#endif
                geometry_msgs::PointStampedConstPtr mocap_msg =
                    m.instantiate<geometry_msgs::PointStamped>();
                if (!mocap_msg) {
                    std::cerr << "Error instantiating point message" << std::endl;
                    continue;  // Skip this message if instantiation fails
                }

                int64_t time = mocap_msg->header.stamp.toNSec();

                point_msgs.push_back(mocap_msg);

                int64_t msg_arrival_time = m.getTime().toNSec();
                system_to_mocap_offset_vec.push_back(time - msg_arrival_time);
#if VERBOSE
                std::cout << "...done" << std::endl;
#endif
            }

            num_msgs++;
        }

        std::cout << "Total number of messages: " << num_msgs << std::endl;

        data->image_timestamps.clear();
        data->image_timestamps.insert(data->image_timestamps.begin(), image_timestamps.begin(), image_timestamps.end());

        data->depth_timestamps.clear();
        data->depth_timestamps.insert(data->depth_timestamps.begin(), depth_timestamps.begin(), depth_timestamps.end());

        if (system_to_mocap_offset_vec.size() > 0) {
            int64_t system_to_imu_offset =
                system_to_imu_offset_vec[system_to_imu_offset_vec.size() / 2];

            int64_t system_to_mocap_offset =
                system_to_mocap_offset_vec[system_to_mocap_offset_vec.size() / 2];

            data->mocap_to_imu_offset_ns =
                system_to_imu_offset - system_to_mocap_offset;
        }

        data->gt_pose_data.clear();
        data->gt_timestamps.clear();

        if (!mocap_msgs.empty())
            for (size_t i = 0; i < mocap_msgs.size() - 1; i++) {
                auto mocap_msg = mocap_msgs[i];

                int64_t time = mocap_msg->header.stamp.toNSec();

                Eigen::Quaterniond q(
                    mocap_msg->transform.rotation.w, mocap_msg->transform.rotation.x,
                    mocap_msg->transform.rotation.y, mocap_msg->transform.rotation.z);

                Eigen::Vector3d t(mocap_msg->transform.translation.x,
                                  mocap_msg->transform.translation.y,
                                  mocap_msg->transform.translation.z);

                int64_t timestamp_ns = time + data->mocap_to_imu_offset_ns;
                data->gt_timestamps.emplace_back(timestamp_ns);
                data->gt_pose_data.emplace_back(q, t);
            }

        if (!point_msgs.empty())
            for (size_t i = 0; i < point_msgs.size() - 1; i++) {
                auto point_msg = point_msgs[i];

                int64_t time = point_msg->header.stamp.toNSec();

                Eigen::Vector3d t(point_msg->point.x, point_msg->point.y,
                                  point_msg->point.z);

                int64_t timestamp_ns = time;  // + data->mocap_to_imu_offset_ns;
                data->gt_timestamps.emplace_back(timestamp_ns);
                data->gt_pose_data.emplace_back(Sophus::SO3d(), t);
            }

        std::cout << "Total number of messages: " << num_msgs << std::endl;
        std::cout << "Image size: " << data->image_data_idx.size() << std::endl;
        std::cout << "Depth size: " << data->depth_data_idx.size() << std::endl;

        std::cout << "Min time: " << min_time << " max time: " << max_time
                  << " mocap to imu offset: " << data->mocap_to_imu_offset_ns
                  << std::endl;

        std::cout << "Number of mocap poses: " << data->gt_timestamps.size()
                  << std::endl;
    }

    void reset() { data.reset(); }

    VioDatasetPtr get_data() {
        // return std::dynamic_pointer_cast<VioDataset>(data);
        return data;
    }

   private:
    std::shared_ptr<RosbagVioDataset> data;
};

}  // namespace slamplay
