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

#include "geometry/TrajectoryAlignment.h"
#include "geometry/TrajectoryUtils.h"
#include "viz/TrajectoryViz.h"

#include <iostream>
#include <random>
#include <vector>

// Function to generate a sinusoidal 3D trajectory
std::vector<Sophus::SE3d> generateSinusoidalTrajectory(int num_points, double amplitude, double frequency) {
    std::vector<Sophus::SE3d> trajectory;
    for (int i = 0; i < num_points; i++) {
        double t = static_cast<double>(i);
        double x = amplitude * std::sin(frequency * t);
        double y = amplitude * std::cos(frequency * t);
        double z = amplitude * std::sin(frequency * t / 2);  // Different frequency for z-axis

        const auto translation = Eigen::Vector3d(x, y, z);
        const auto pose = Sophus::SE3d(Eigen::Matrix3d::Identity(), translation);
        trajectory.push_back(pose);
    }

    // Add rotation to each pose, aligning the z-axis with the delta translation
    for (int i = 1; i < num_points; i++) {
        const auto& prev_pose = trajectory[i - 1];
        const auto& curr_pose = trajectory[i];

        const auto prev_translation = prev_pose.translation();
        const auto curr_translation = curr_pose.translation();

        const auto prev_to_curr = curr_translation - prev_translation;

        // Normalize the direction vector to get the unit vector of translation direction
        Eigen::Vector3d z_direction = prev_to_curr.normalized();

        // Compute the rotation matrix to align z-axis of the pose with the delta translation
        Eigen::Vector3d z_axis(0, 0, 1);  // Default Z-axis (the original pose's Z-axis)
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(z_axis, z_direction);

        // Apply the computed rotation
        // const auto rotation = Eigen::Quaterniond(prev_pose.rotationMatrix()) * q;
        const auto rotation = q;

        // Set the new pose with the adjusted rotation and current translation
        trajectory[i] = Sophus::SE3d(rotation, curr_translation);
    }

    return trajectory;
}

// Main function to run the test
int main() {
    // Parameters for the sinusoidal trajectory
    int num_points = 500;
    double amplitude = 10.0;
    double frequency = 0.1;

    // Generate a reference sinusoidal trajectory
    auto reference_trajectory = generateSinusoidalTrajectory(num_points, amplitude, frequency);

    // Add noise to the reference trajectory to simulate the noisy trajectory
    double noise_level = 0.001;
    auto noisy_trajectory = slamplay::addNoise(reference_trajectory, noise_level);

    constexpr double deg_to_rad = M_PI / 180.0;
    constexpr double rad_to_deg = 180.0 / M_PI;

    // Transform the noisy trajectory to align it with the reference trajectory
    double roll = 10 * deg_to_rad;
    double pitch = -2 * deg_to_rad;
    double yaw = 5 * deg_to_rad;
    double tx = 0.1;
    double ty = -0.05;
    double tz = -0.2;

    std::cout << "Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << ", tx: " << tx << ", ty: " << ty << ", tz: " << tz << std::endl;

    const auto TcorrGt = slamplay::fromRPYAndTranslation(roll, pitch, yaw, tx, ty, tz);
    noisy_trajectory = slamplay::transform(TcorrGt, noisy_trajectory);

    // Create timestamps for the reference and noisy trajectories
    std::vector<int64_t> gt_t_ns(num_points), filter_t_ns(num_points);
    const int64_t Ts_ns = 30 * 1000000;  // 30 ms
    for (int i = 0; i < num_points; i++) {
        gt_t_ns[i] = filter_t_ns[i] = Ts_ns * i;
    }

    // Align the noisy trajectory to the reference using alignSVD
    auto [error, TcorrEstimated] = slamplay::alignSVD(filter_t_ns, noisy_trajectory, gt_t_ns, reference_trajectory);

    // Get a corrected trajectory
    auto corrected_trajectory = slamplay::transform(TcorrEstimated, noisy_trajectory);

    // Print the final alignment error
    std::cout << "Final alignment error: " << error << std::endl;

    auto Terror = TcorrEstimated * TcorrGt;
    std::cout << "Terror: " << Terror.matrix() << std::endl;
    auto [roll_e, pitch_e, yaw_e, tx_e, ty_e, tz_e] = slamplay::getRPYAndTranslation(Terror);
    std::cout << "Estimation error: roll_e: " << roll_e * rad_to_deg << "[degs], pitch_e: " << pitch_e * rad_to_deg << "[degs], yaw_e: " << yaw_e * rad_to_deg << "[degs], tx_e: " << tx_e << ", ty_e: " << ty_e << ", tz_e: " << tz_e << std::endl;

    // Visualize the reference and noisy trajectories
    slamplay::TrajectoryViz viz;
    viz.start();
    viz.addTrajectory(reference_trajectory, slamplay::Trajectory::Color(0, 255, 0), "reference");
    viz.addTrajectory(noisy_trajectory, slamplay::Trajectory::Color(255, 0, 0), "noisy");
    viz.addTrajectory(corrected_trajectory, slamplay::Trajectory::Color(0, 0, 255), "corrected");

    while (viz.isRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}
