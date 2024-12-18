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

#include "myslam/common_include.h"

namespace myslam {

struct IMUData {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUData(const double &gx, const double &gy, const double &gz,
            const double &ax, const double &ay, const double &az,
            const double &t) :
            mfGyro(gx, gy, gz), mfAcce(ax, ay, az), mfTimeStamp(t) {}

    IMUData(const Eigen::Vector3d gyro, const Eigen::Vector3d &acce, const double &time)
            : mfGyro(gyro), mfAcce(acce), mfTimeStamp(time) {}

    //Raw data of imu
    Eigen::Vector3d mfGyro;//gyr data
    Eigen::Vector3d mfAcce;//acc data
    double mfTimeStamp;//timestamp

    //covariance of measurement
    static Eigen::Matrix3d mfGyrMeasCov;//The covariance matrix of the gyroscope is a diagonal matrix composed of variances
    static Eigen::Matrix3d mfAccMeasCov;//The covariance matrix of the accelerometer is a diagonal matrix composed of variances

    //covariance of bias random walk, RW stands for random walk
    static Eigen::Matrix3d mfGyrBiasRWCov;//covariance matrix of random walk
    static Eigen::Matrix3d mfAccBiasRWCov;//covariance matrix of accelerometer random walk

};

typedef std::vector<IMUData, Eigen::aligned_allocator<IMUData> > VecIMU;
}

