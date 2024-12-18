// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#pragma once

#include "datasets/DatasetIo.h"
#include "io/filesystem.h"

#include <opencv2/highgui/highgui.hpp>

namespace slamplay {

class TumRgbdDataset : public VioDataset {
    size_t num_cams;

    std::string path;

    std::vector<int64_t> image_timestamps;
    std::unordered_map<int64_t, std::string> image_path;

    std::vector<int64_t> depth_timestamps;
    std::unordered_map<int64_t, std::string> depth_path;

    // vector of images for every timestamp
    // assumes vectors size is num_cams for every timestamp with null pointers for
    // missing frames
    // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

    Eigen::aligned_vector<AccelData> accel_data;
    Eigen::aligned_vector<GyroData> gyro_data;

    std::vector<int64_t> gt_timestamps;                // ordered gt timestamps
    Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;  // TODO: change to eigen aligned

    int64_t mocap_to_imu_offset_ns;

   public:
    ~TumRgbdDataset() {};

    size_t get_num_cams() const { return num_cams; }

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
        const std::vector<std::string> folder = {"/"};

        for (size_t i = 0; i < num_cams; i++) {
            std::string full_image_path = path + folder[0] + image_path[t_ns];

            if (fs::exists(full_image_path)) {
                cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);
                if (!img.empty()) {
                    res[i].img = img;
                } else {
                    std::cerr << "Failed to load RGB image: " << full_image_path << std::endl;
                }
            }
        }
        return res;
    }

    std::vector<DepthData> get_depth_data(int64_t t_ns) {
        std::vector<DepthData> res(num_cams);
        const std::vector<std::string> folder = {"/"};

        for (size_t i = 0; i < num_cams; i++) {
            std::string full_depth_path = path + folder[0] + depth_path[t_ns];

            if (fs::exists(full_depth_path)) {
                cv::Mat depth_img = cv::imread(full_depth_path, cv::IMREAD_UNCHANGED);
                if (!depth_img.empty()) {
                    DepthData depth_data;
                    depth_data.img = depth_img;
                    res[i] = depth_data;
                } else {
                    std::cerr << "Failed to load depth image at " << full_depth_path << std::endl;
                }
            } else {
                std::cerr << "Depth image not found: " << full_depth_path << std::endl;
            }
        }

        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class TumIO;
};

class TumIO : public DatasetIoInterface {
   public:
    TumIO() {}

    void read(const std::string &path) {
        if (!fs::exists(path))
            std::cerr << "No dataset found in " << path << std::endl;

        data.reset(new TumRgbdDataset);

        data->num_cams = 1;
        data->path = path;

        // read_image_timestamps(path + "/rgb.txt");
        // read_depth_timestamps(path + "/depth.txt");
        read_associations(path + "/associations.txt");

        if (fs::exists(path + "/groundtruth.txt")) {
            read_gt_data_pose(path + "/groundtruth.txt");
        }
    }

    void reset() { data.reset(); }

    VioDatasetPtr get_data() { return data; }

   private:
    void read_image_timestamps(const std::string &path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line[0] == '#') continue;
            std::stringstream ss(line);
            char tmp;
            double t_s;
            std::string path;
            ss >> t_s >> tmp >> path;
            int64_t t_ns = t_s * 1e9;

            data->image_timestamps.emplace_back(t_ns);
            data->image_path[t_ns] = path;
        }
    }

    void read_depth_timestamps(const std::string &path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line[0] == '#') continue;
            std::stringstream ss(line);
            char tmp;
            double t_s;
            std::string path;
            ss >> t_s >> tmp >> path;
            int64_t t_ns = t_s * 1e9;

            data->depth_timestamps.emplace_back(t_ns);
            data->depth_path[t_ns] = path;
        }
    }

    void read_associations(const std::string &path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line[0] == '#') continue;
            std::stringstream ss(line);
            double t_s_rgb, t_s_depth;
            std::string path_rgb;
            std::string path_depth;
            ss >> t_s_rgb >> path_rgb >> t_s_depth >> path_depth;
            int64_t t_ns_rgb = t_s_rgb * 1e9;
            int64_t t_ns_depth = t_s_depth * 1e9;

#if 0
            std::cout << "t_ns_rgb: " << t_ns_rgb << " path_rgb: " << path_rgb << " t_ns_depth: " << t_ns_depth << " path_depth: " << path_depth << std::endl;
#endif
            data->image_timestamps.emplace_back(t_ns_rgb);
            data->image_path[t_ns_rgb] = path_rgb;
            data->depth_timestamps.emplace_back(t_ns_depth);
            data->depth_path[t_ns_depth] = path_depth;
        }
    }

    void read_gt_data_pose(const std::string &filepath) {
        data->gt_timestamps.clear();
        data->gt_pose_data.clear();

        int i = 0;
        std::ifstream f(filepath);
        std::string line;

        while (std::getline(f, line)) {
            if (line[0] == '#') continue;
            std::stringstream ss(line);

            char tmp;
            double t_s;
            Eigen::Quaterniond q;
            Eigen::Vector3d pos;

            // timestamp tx ty tz qx qy qz qw
            ss >> t_s >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >> tmp >> q.coeffs()[1] >> tmp >> q.coeffs()[2] >> tmp >> q.coeffs()[3] >> tmp >> q.coeffs()[0];
#if 0
            std::cout << "timestamp: " << t_s
                      << ", pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                      << ", q: " << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << std::endl;
#endif
            int64_t t_ns = t_s * 1e9;
            data->gt_timestamps.emplace_back(t_ns);
            data->gt_pose_data.emplace_back(q, pos);
            i++;
        }
    }

    std::shared_ptr<TumRgbdDataset> data;
};

}  // namespace slamplay
