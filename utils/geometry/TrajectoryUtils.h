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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <vector>

#include "sophus/se3.hpp"

namespace slamplay {

// Function to add noise to the trajectory
template <typename T>
inline std::vector<Sophus::SE3<T>> addNoise(const std::vector<Sophus::SE3<T>>& trajectory, double noise_level) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, noise_level);

    Eigen::aligned_vector<Sophus::SE3<T>> noisy_trajectory;
    for (const auto& pose : trajectory) {
        auto noisy_point = pose.translation();
        noisy_point.x() += dis(gen);
        noisy_point.y() += dis(gen);
        noisy_point.z() += dis(gen);
        const auto noisy_pose = Sophus::SE3d(pose.rotationMatrix(), noisy_point);
        noisy_trajectory.push_back(noisy_pose);
    }
    return noisy_trajectory;
}

template <typename T>
inline Sophus::SE3<T> fromRPYAndTranslation(const T& roll, const T& pitch, const T& yaw, const T& tx, const T& ty, const T& tz) {
    const auto rotation = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    const auto translation = Eigen::Vector3d(tx, ty, tz);
    return Sophus::SE3<T>(rotation, translation);
}

template <typename T>
inline std::tuple<T, T, T, T, T, T> getRPYAndTranslation(const Sophus::SE3<T>& transformation) {
    const auto rotation = transformation.rotationMatrix();
    const auto roll = std::atan2(rotation(2, 1), rotation(2, 2));
    const auto pitch = std::atan2(-rotation(2, 0), std::sqrt(rotation(2, 1) * rotation(2, 1) + rotation(2, 2) * rotation(2, 2)));
    const auto yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    const auto translation = transformation.translation();
    return std::make_tuple(roll, pitch, yaw, translation.x(), translation.y(), translation.z());
}

template <typename T>
inline std::vector<Sophus::SE3<T>> transform(const Sophus::SE3<T>& transformation, const std::vector<Sophus::SE3<T>>& trajectory) {
    std::vector<Sophus::SE3<T>> out;
    out.reserve(trajectory.size());
    for (auto& pose : trajectory) {
        out.push_back(transformation * pose);
    }
    return out;
}

template <typename T>
inline std::vector<Sophus::SE3<T>> transform(const double roll, const double pitch, const double yaw, const double tx, const double ty, const double tz, const std::vector<Sophus::SE3<T>>& trajectory) {
    const auto transformation = fromRPYAndTranslation(roll, pitch, yaw, tx, ty, tz);
    return transform(transformation, trajectory);
}

}  // namespace slamplay