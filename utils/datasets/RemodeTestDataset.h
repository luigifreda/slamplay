#pragma once

#include <fstream>     
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

bool readRemodeDatasetFiles(
    const std::string &path,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth,
    int width=640,
    int height=480) 
{
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: image file name tx, ty, tz, qx, qy, qz, qw, note that it is TWC instead of TCW
        std::string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + std::string("/images/") + image);
        poses.push_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                 Eigen::Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}