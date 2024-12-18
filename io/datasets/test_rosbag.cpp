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
#include "datasets/DatasetIo.h"
#include "image/image_depth.h"
#include "viz/TrajectoryViz.h"

#include <opencv2/opencv.hpp>
#include <string>
#include "macros.h"

using namespace std;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

void replayImageAndDepthSynchronized(slamplay::VioDatasetPtr& dataset) {
    // Check if image and depth timestamps are close enough for synchronization
    constexpr int64_t sync_threshold_ns = 5e6;  // milliseconds

    const std::vector<int64_t>& image_timestamps = dataset->get_image_timestamps();
    const std::vector<int64_t>& depth_timestamps = dataset->get_depth_timestamps();

    auto image_it = image_timestamps.begin();
    auto depth_it = depth_timestamps.begin();

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Depth", cv::WINDOW_NORMAL);

    // Loop through timestamps, synchronizing image and depth data
    while (image_it != image_timestamps.end() && depth_it != depth_timestamps.end()) {
        int64_t image_time = *image_it;
        int64_t depth_time = *depth_it;

        if (std::abs(image_time - depth_time) < sync_threshold_ns) {
            auto image_data = dataset->get_image_data(image_time);
            auto depth_data = dataset->get_depth_data(depth_time);

            // Assuming only one camera and one depth sensor
            if (!image_data.empty() && !depth_data.empty()) {
                cv::Mat image = image_data[0].img;
                cv::Mat depth = depth_data[0].img;

                // Display image and depth
                cv::imshow("Image", image);
                slamplay::showDepthImage("Depth", depth);
                cv::waitKey(10);
            }

            // Increment both iterators
            ++image_it;
            ++depth_it;
        } else if (image_time < depth_time) {
            // Image timestamp is earlier, increment image iterator
            ++image_it;
        } else {
            // Depth timestamp is earlier, increment depth iterator
            ++depth_it;
        }
    }

    cv::destroyAllWindows();
}

void replayImageUnsynchronized(slamplay::VioDatasetPtr& dataset) {
    for (size_t i = 0; i < dataset->get_image_timestamps().size(); i++) {
        int64_t t_img_ns = dataset->get_image_timestamps()[i];
        auto img_data = dataset->get_image_data(t_img_ns);

        cv::Mat& image = img_data[0].img;
        if (image.empty()) {
            std::cout << "empty image at timestamp " << t_img_ns << "" << std::endl;
            continue;
        }

        cv::imshow("image", image);
        cv::waitKey(10);
    }

    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    std::string dataset_type = "bag";

    std::string dataset_path = "/home/luigi/Work/datasets/rgbd_datasets/tum/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.bag";
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <dataset_path>" << endl;
    } else if (argc == 2) {
        dataset_path = argv[1];
    }

    std::cout << "Reading " << dataset_type << " dataset: " << dataset_path << std::endl;

    slamplay::DatasetIoInterfacePtr dataset_io = slamplay::DatasetIoFactory::getDatasetIo(dataset_type);

    dataset_io->read(dataset_path);

    slamplay::VioDatasetPtr dataset = dataset_io->get_data();

    std::cout << "Found:" << std::endl
              << "\t " << dataset->get_image_timestamps().size() << " RGB images" << std::endl
              << "\t " << dataset->get_depth_timestamps().size() << " depth images" << std::endl
              << "\t " << dataset->get_accel_data().size() << " accel data" << std::endl
              << "\t " << dataset->get_gyro_data().size() << " gyro data" << std::endl;

    auto gt_trajectory = dataset->get_gt_pose_data();

    slamplay::TrajectoryViz viz;
    if (!gt_trajectory.empty()) {
        viz.setDownsampleCameraVizFactor(10);
        viz.start();
        viz.addTrajectory(gt_trajectory, slamplay::Trajectory::Color(0, 255, 0), "gt");
    }

    // replayImageUnsynchronized(dataset);
    replayImageAndDepthSynchronized(dataset);

    std::cout << "done!" << std::endl;
    return 0;
}
