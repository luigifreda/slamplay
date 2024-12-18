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

int main(int argc, char **argv) {
    std::string dataset_type = "replica";

    std::string dataset_path = "/media/luigi/T7/datasets/rgbd_datasets/replica/room2";
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
              << "\t " << dataset->get_depth_timestamps().size() << " depth images" << std::endl;

    auto gt_trajectory = dataset->get_gt_pose_data();

    slamplay::TrajectoryViz viz;
    viz.setDownsampleCameraVizFactor(30);
    viz.start();
    viz.setTrajectory(gt_trajectory);

    for (size_t i = 0; i < dataset->get_image_timestamps().size(); i++) {
        int64_t t_img_ns = dataset->get_image_timestamps()[i];
        auto img_data = dataset->get_image_data(t_img_ns);

        int64_t t_depth_ns = dataset->get_depth_timestamps()[i];
        auto depth_data = dataset->get_depth_data(t_depth_ns);

        cv::Mat img = img_data[0].img;
        cv::Mat depth = depth_data[0].img;

        cv::imshow("image", img);
        // cv::imshow("depth", depth);
        slamplay::showDepthImage("depth", depth);
        cv::waitKey(1);
    }

    std::cout << "done!" << std::endl;
    return 0;
}
