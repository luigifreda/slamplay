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

