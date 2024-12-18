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

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_CEREAL
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#endif

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include "sophus/SophusUtils.h"

namespace slamplay {

struct ImageData {
    ImageData() : exposure(0) {}

    // ManagedImage<uint16_t>::Ptr img;
    cv::Mat img;
    double exposure;
};

struct DepthData {
    cv::Mat img;
};

struct Observations {
    Eigen::aligned_vector<Eigen::Vector2d> pos;
    std::vector<int> id;
};

struct GyroData {
    int64_t timestamp_ns;
    Eigen::Vector3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AccelData {
    int64_t timestamp_ns;
    Eigen::Vector3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseData {
    int64_t timestamp_ns;
    Sophus::SE3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MocapPoseData {
    int64_t timestamp_ns;
    Sophus::SE3d data;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AprilgridCornersData {
    int64_t timestamp_ns;
    int cam_id;

    Eigen::aligned_vector<Eigen::Vector2d> corner_pos;
    std::vector<int> corner_id;
};

class VioDataset {
   public:
    virtual ~VioDataset() {};

    virtual size_t get_num_cams() const = 0;

    virtual std::vector<int64_t> &get_image_timestamps() = 0;
    virtual std::vector<int64_t> &get_depth_timestamps() = 0;

    virtual const Eigen::aligned_vector<AccelData> &get_accel_data() const = 0;
    virtual const Eigen::aligned_vector<GyroData> &get_gyro_data() const = 0;
    virtual const std::vector<int64_t> &get_gt_timestamps() const = 0;
    virtual const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const = 0;
    virtual int64_t get_mocap_to_imu_offset_ns() const = 0;
    virtual std::vector<ImageData> get_image_data(int64_t t_ns) = 0;
    virtual std::vector<DepthData> get_depth_data(int64_t t_ns) = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::shared_ptr<VioDataset> VioDatasetPtr;

class DatasetIoInterface {
   public:
    virtual void read(const std::string &path) = 0;
    virtual void reset() = 0;
    virtual VioDatasetPtr get_data() = 0;

    virtual ~DatasetIoInterface() {};
};

typedef std::shared_ptr<DatasetIoInterface> DatasetIoInterfacePtr;

class DatasetIoFactory {
   public:
    // Supported dataset types:
    // euroc, bag, uzh, kitti, tum_rgbd, tum_vio (see DatasetIo.cpp)
    static DatasetIoInterfacePtr getDatasetIo(const std::string &dataset_type,
                                              bool load_mocap_as_gt = false);
};

}  // namespace slamplay

#ifdef USE_CEREAL
namespace cereal {

template <class Archive>
void serialize(Archive &archive, basalt::ManagedImage<uint8_t> &m) {
    archive(m.w);
    archive(m.h);

    m.Reinitialise(m.w, m.h);

    archive(binary_data(m.ptr, m.size()));
}

template <class Archive>
void serialize(Archive &ar, basalt::GyroData &c) {
    ar(c.timestamp_ns, c.data);
}

template <class Archive>
void serialize(Archive &ar, basalt::AccelData &c) {
    ar(c.timestamp_ns, c.data);
}

}  // namespace cereal
#endif