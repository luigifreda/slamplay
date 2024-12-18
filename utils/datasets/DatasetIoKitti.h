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

#include "datasets/DatasetIo.h"
#include "io/filesystem.h"

#include <opencv2/highgui/highgui.hpp>

namespace slamplay {

class KittiVioDataset : public VioDataset {
    size_t num_cams;

    std::string path;

    std::vector<int64_t> image_timestamps;
    std::unordered_map<int64_t, std::string> image_path;

    std::vector<int64_t> depth_timestamps;  // empty

    // vector of images for every timestamp
    // assumes vectors size is num_cams for every timestamp with null pointers for
    // missing frames
    // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

    Eigen::aligned_vector<AccelData> accel_data;
    Eigen::aligned_vector<GyroData> gyro_data;

    std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
    Eigen::aligned_vector<Sophus::SE3d>
        gt_pose_data;  // TODO: change to eigen aligned

    int64_t mocap_to_imu_offset_ns;

   public:
    ~KittiVioDataset() {};

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

        for (size_t i = 0; i < num_cams; i++) {
            std::string full_image_path = path + image_path[t_ns];

            if (fs::exists(full_image_path)) {
                cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);
                res[i].img = img;
            }
        }

        return res;
    }

    std::vector<DepthData> get_depth_data(int64_t t_ns) {
        std::vector<DepthData> res(num_cams);
        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class KittiIO;
    bool is_color = false;
};

class KittiIO : public DatasetIoInterface {
   public:
    KittiIO() {}

    void read(const std::string &path) {
        if (!fs::exists(path))
            std::cerr << "No dataset found in " << path << std::endl;

        data.reset(new KittiVioDataset);
        data->is_color = is_color(path);

        data->num_cams = 2;
        data->path = path;

        std::string sequence_name = path.substr(path.find_last_of("/") + 1);

        read_image_timestamps(path + "/times.txt");
        read_image_names(path, data->is_color);

        if (fs::exists(path + "/poses.txt")) {
            read_gt_data_pose(path + "/poses.txt");
        } else if (fs::exists(path + "/" + sequence_name + ".txt")) {
            read_gt_data_pose(path + "/" + sequence_name + ".txt");
        } else {
            std::cerr << "No ground truth data found in " << path << std::endl;
        }
    }

    void reset() { data.reset(); }

    VioDatasetPtr get_data() { return data; }

   private:
    std::string generate_filename(size_t number) {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << number;  // 6-digit zero-padded integer
        return oss.str();
    }

    void read_image_timestamps(const std::string &path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line[0] == '#') continue;
            std::stringstream ss(line);
            double t_s;
            ss >> t_s;
            int64_t t_ns = t_s * 1e9;
            data->image_timestamps.emplace_back(t_ns);
        }
    }

    void read_image_names(const std::string &path, bool is_color) {
        const std::vector<std::string> folder = {"/image_0/", "/image_1/"};
        const std::vector<std::string> folder_color = {"/image_2/", "/image_3/"};

        const size_t expected_num_files = data->image_timestamps.size();

        for (size_t i = 0; i < data->num_cams; i++) {
            const std::string folder_i = is_color ? folder_color[i] : folder[i];
            std::string full_folder_path = path + folder_i;

            // number of files in the path
            size_t num_files = 0;
            if (fs::exists(full_folder_path) && fs::is_directory(full_folder_path)) {
                for (const auto &entry : fs::directory_iterator(full_folder_path)) {
                    if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                        num_files++;
                    }
                }
            } else {
                std::cerr << "Invalid folder path: " << full_folder_path << std::endl;
            }

            if (num_files != expected_num_files) {
                std::cerr << "Number of files in " << full_folder_path << " is " << num_files
                          << " but expected " << expected_num_files << std::endl;
                continue;
            }
            for (size_t jj = 0; jj < expected_num_files; jj++) {
                std::string filename = generate_filename(jj);
                data->image_path[data->image_timestamps[jj]] = folder_i + filename + ".png";
            }
        }
    }

    void read_gt_data_pose(const std::string &path) {
        data->gt_timestamps.clear();
        data->gt_pose_data.clear();

        int i = 0;

        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line[0] == '#') continue;

            std::stringstream ss(line);

            Eigen::Matrix3d rot;
            Eigen::Vector3d pos;

            ss >> rot(0, 0) >> rot(0, 1) >> rot(0, 2) >> pos[0] >> rot(1, 0) >>
                rot(1, 1) >> rot(1, 2) >> pos[1] >> rot(2, 0) >> rot(2, 1) >>
                rot(2, 2) >> pos[2];

            data->gt_timestamps.emplace_back(data->image_timestamps[i]);
            data->gt_pose_data.emplace_back(Eigen::Quaterniond(rot), pos);
            i++;
        }
    }

    bool is_color(const std::string &path) {
        std::string color_left_image_path = path + "/image_2/";
        std::string color_right_image_path = path + "/image_3/";
        if (fs::exists(color_left_image_path) && fs::exists(color_right_image_path)) {
            return true;
        }
        return false;
    }

    std::shared_ptr<KittiVioDataset> data;
};

}  // namespace slamplay
