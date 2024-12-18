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
//
// Created by gaoxiang on 19-5-4.
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay
//
#pragma once

#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/imudata.h"

#include <map>

namespace myslam {

enum class DatasetType : int { KITTI = 0,
                               EUROC };

/**
 * Dataset read
 * The configuration file path is passed in during construction, and the dataset_dir of the configuration file is the dataset path
 * After Init, the camera and the next frame image can be obtained
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string &dataset_path, bool use_half_resolution = false);

    virtual bool Init() = 0;
    virtual Frame::Ptr NextFrame() = 0;

    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

    int GetCurrentFrameIndex() const {
        return current_image_index_;
    }

   protected:
    std::string dataset_path_;
    int current_image_index_ = 0;
    bool end_ = false;
    bool use_half_resolution_ = false;

    std::vector<Camera::Ptr> cameras_;
};

class DatasetKitti : public Dataset {
   public:
    DatasetKitti(const std::string &dataset_path, bool use_half_resolution = false) : Dataset(dataset_path, use_half_resolution) {}

    bool Init() override;
    Frame::Ptr NextFrame() override;
};

class DatasetEuroc : public Dataset {
   public:
    typedef std::map<double, Sophus::SE3d, std::less<double>> TrajectoryType;

    DatasetEuroc(const std::string &dataset_path, bool use_half_resolution = false) : Dataset(dataset_path, use_half_resolution) {}

    bool Init() override;
    Frame::Ptr NextFrame() override;

   protected:
    // Load the stereo image data
    // Input: left eye image directory, right eye image directory, timestamp file
    // Output: left-eye image file path, right-eye image file path, timestamp after sorting
    bool LoadImages(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                    std::vector<std::string> &vstrImageLeft,
                    std::vector<std::string> &vstrImageRight,
                    std::vector<double> &vTimeStamps);

    // Load the IMU data
    bool LoadImus(const std::string &strImuPath, VecIMU &vImus);

    // Load the ground truth trajectory
    bool LoadGroundTruthTraj(const std::string &trajPath, TrajectoryType &trajectory);

   protected:
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;
    std::vector<double> vTimeStamp;
    VecIMU vimus;
    TrajectoryType ground_truth;

    // undistort rectification maps
    cv::Mat M1l, M2l, M1r, M2r;
};

}  // namespace myslam
