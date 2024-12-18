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

#include "myslam/dataset.h"
#include "myslam/frame.h"
#include "myslam/config.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;


namespace myslam {

Dataset::Dataset(const std::string& dataset_path, bool use_half_resolution)
    : dataset_path_(dataset_path), use_half_resolution_(use_half_resolution) {}


// ===============================

bool DatasetKitti::Init() {
    // read camera intrinsics and extrinsics
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
             projection_data[4], projection_data[5], projection_data[6],
             projection_data[8], projection_data[9], projection_data[10];
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * t;
        if(use_half_resolution_) K = K * 0.5;    
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    return true;
}

Frame::Ptr DatasetKitti::NextFrame() {
    if(end_) return nullptr; 

    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;

    // read images
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(ERROR) << "cannot find images at index " << current_image_index_;
        if(current_image_index_>0)  LOG(INFO) << "dataset end"; 
        end_ = true; 
        return nullptr;
    }

    auto new_frame = Frame::CreateFrame();
    if(use_half_resolution_)
    {
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;    
    }           
    else
    {
        new_frame->left_img_ = image_left;
        new_frame->right_img_ = image_right;       
    } 

    current_image_index_++;
    return new_frame;
}


// ===============================

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

bool DatasetEuroc::Init() {
    // read camera intrinsics and extrinsics
    std::string timestamps_file_path = Config::Get<std::string>("timestamps_path"); 
    std::string images_left_dir = dataset_path_ + "/cam0/data"; 
    std::string images_right_dir = dataset_path_ + "/cam1/data"; 
    std::string imu_file_path = dataset_path_ + "/imu0/data.csv"; 

    LOG(INFO) << "loading images ...";
    if(LoadImages(images_left_dir, images_right_dir, timestamps_file_path, 
                  vstrImageLeft, vstrImageRight, vTimeStamp) == false)
        return false;
    
    LOG(INFO) << "loading imu ...";    
    if (LoadImus(imu_file_path, vimus) == false)
        return 1;

    // rectification parameters
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;    

    auto& file_storage = Config::File();
    file_storage["LEFT.K"] >> K_l;
    file_storage["RIGHT.K"] >> K_r;

    file_storage["LEFT.P"] >> P_l;
    file_storage["RIGHT.P"] >> P_r;

    file_storage["LEFT.R"] >> R_l;
    file_storage["RIGHT.R"] >> R_r;

    file_storage["LEFT.D"] >> D_l;
    file_storage["RIGHT.D"] >> D_r;

    int rows_l = file_storage["LEFT.height"];
    int cols_l = file_storage["LEFT.width"];
    int rows_r = file_storage["RIGHT.height"];
    int cols_r = file_storage["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || 
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
        LOG(FATAL) << "Calibration parameters to rectify stereo are missing!" << endl;
        return 1;
    }

    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, 
                                M1l, M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, 
                                M1r, M2r);
   
    // rectified camera params 
    float fx = Config::Get<float>("Camera.fx");
    float fy = Config::Get<float>("Camera.fy");
    float cx = Config::Get<float>("Camera.cx");
    float cy = Config::Get<float>("Camera.cy");
    const float bf = Config::Get<float>("Camera.bf");

    const float baseline = bf/fx; 
    Eigen::Vector3d tl(0,0,0); 
    Eigen::Vector3d tr(-baseline,0,0);   

    if(use_half_resolution_) {
        fx *= 0.5; 
        fy *= 0.5; 
        cx *= 0.5; 
        cy *= 0.5;                 
    }
  
    Camera::Ptr camera_left(new Camera(fx, fy, cx, cy, 0, SE3(SO3(), tl))); 
    Camera::Ptr camera_right(new Camera(fx, fy, cx, cy, baseline, SE3(SO3(), tr))); 
#if 0     
    std::cout << "camera_left " << std::endl; 
    std::cout << "          K:" << camera_left->K() << std::endl;
    std::cout << "          pose:" << camera_left->pose().matrix3x4() << std::endl;        
    std::cout << "camera_right " << std::endl; 
    std::cout << "          K:" << camera_right->K() << std::endl;
    std::cout << "          pose:" << camera_right->pose().matrix3x4() << std::endl;                                              
#endif     
    cameras_.push_back(camera_left);
    cameras_.push_back(camera_right);    

    LOG(INFO) << "init done ..."; 

    return true;
}


Frame::Ptr DatasetEuroc::NextFrame() { 
    if(end_) return nullptr; 
    cv::Mat image_left, image_left_rect, image_right, image_right_rect;

    // read images
    image_left = cv::imread(vstrImageLeft[current_image_index_], cv::IMREAD_UNCHANGED);
    image_right = cv::imread(vstrImageRight[current_image_index_], cv::IMREAD_UNCHANGED);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(ERROR) << "cannot find images at index " << current_image_index_;
        if(current_image_index_>0)  LOG(INFO) << "dataset end"; 
        end_ = true; 
        return nullptr;
    }

    cv::remap(image_left, image_left_rect, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(image_right, image_right_rect, M1r, M2r, cv::INTER_LINEAR);    

    auto new_frame = Frame::CreateFrame();
    if(use_half_resolution_)
    {
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left_rect, image_left_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);
        cv::resize(image_right_rect, image_right_resized, cv::Size(), 0.5, 0.5,
                cv::INTER_NEAREST);
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;    
    }           
    else
    {
        new_frame->left_img_ = image_left_rect;
        new_frame->right_img_ = image_right_rect;       
    } 

    current_image_index_++;
    return new_frame;
}

bool DatasetEuroc::LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    if (!fTimes) {
        LOG(ERROR) << "cannot find timestamp file: " << strPathTimes << endl;
        return false;
    }
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);

    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9);
        }
    }
    fTimes.close();

    if (strPathLeft.empty()) {
        LOG(ERROR) << "No images in left folder!" << endl;
        return false;
    }

    if (strPathRight.empty()) {
        LOG(ERROR) << "No images in right folder!" << endl;
        return false;
    }

    LOG(INFO) << "read " << vstrImageLeft.size() << " images";
    return true;
}

bool DatasetEuroc::LoadImus(const string &strImuPath, VecIMU &vImus) {
    ifstream fImus(strImuPath);

    if (!fImus) {
        LOG(ERROR) << "cannot find IMU file: " << strImuPath << endl;
        return false;
    }

    vImus.reserve(30000);
    //int testcnt = 10;
    while (!fImus.eof()) {
        string s;
        getline(fImus, s);
        if (!s.empty()) {
            char c = s.at(0);

            if (c < '0' || c > '9') // skip the comment
                continue;

            stringstream ss;
            ss << s;
            double tmpd;
            int cnt = 0;
            double data[10];    // timestamp, wx,wy,wz, ax,ay,az
            while (ss >> tmpd) {
                data[cnt] = tmpd;
                cnt++;
                if (cnt == 7)
                    break;
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }
            data[0] *= 1e-9;
            IMUData imudata(data[1], data[2], data[3],
                                    data[4], data[5], data[6], data[0]);
            vImus.push_back(imudata);
        }
    }
    fImus.close();

    LOG(INFO) << "read " << vImus.size() << " IMU data";    
    return true;
}

bool DatasetEuroc::LoadGroundTruthTraj(const string &trajPath,
                                       TrajectoryType &trajectory) {

    ifstream fTraj(trajPath);
    if (!fTraj) {
        LOG(ERROR) << "cannot find trajectory file!" << endl;
        return false;
    }

    while (!fTraj.eof()) {
        string s;
        getline(fTraj, s);
        if (!s.empty()) {
            if (s[0] < '0' || s[0] > '9') // not a number
                continue;

            stringstream ss;
            ss << s;
            double timestamp = 0;
            ss >> timestamp;
            ss.ignore();

            timestamp *= 1e-9;

            double data[7];
            for (double &d:data) {
                ss >> d;
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }

            // x,y,z,qw,qx,qy,qz
            Sophus::SE3d pose(Sophus::SO3d(Eigen::Quaterniond(data[3], data[4], data[5], data[6])),
                                           Eigen::Vector3d(data[0], data[1], data[2]));
            trajectory[timestamp] = pose;
        }
    }

    fTraj.close();

    return true;
}


}  // namespace myslam