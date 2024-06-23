//
// Created by haoyuefan on 23-1-16.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <algorithm>

static bool GetFileNames(const std::string &path, std::vector<std::string> &filenames) {
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str()))) {
        std::cerr << "Current folder doesn't exist!" << std::endl;
        return false;
    }
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(filenames.begin(), filenames.end());
    return true;
}

static bool FileExists(const std::string& file) {
    struct stat file_status{};
    if (stat(file.c_str(), &file_status) == 0 &&
        (file_status.st_mode & S_IFREG)) {
        return true;
    }
    return false;
}

static void ConcatenateFolderAndFileNameBase(
        const std::string& folder, const std::string& file_name,
        std::string* path) {
    *path = folder;
    if (path->back() != '/') {
        *path += '/';
    }
    *path = *path + file_name;
}

static std::string ConcatenateFolderAndFileName(
        const std::string& folder, const std::string& file_name) {
    std::string path;
    ConcatenateFolderAndFileNameBase(folder, file_name, &path);
    return path;
}

static void VisualizeMatching(const cv::Mat &image0, const std::vector<cv::KeyPoint> &keypoints0, const cv::Mat &image1,
                              const std::vector<cv::KeyPoint> &keypoints1,
                              const std::vector<cv::DMatch> &superglue_matches, cv::Mat &output_image, double cost_time = -1) {
    if(image0.size != image1.size) return;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, output_image, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    double sc = std::min(image0.rows / 640., 2.0);
    int ht = int(30 * sc);
    std::string title_str = "SuperPoint SuperGlue TensorRT";
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string feature_points_str = "Keypoints: " + std::to_string(keypoints0.size()) + ":" + std::to_string(keypoints1.size());
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string match_points_str = "Matches: " + std::to_string(superglue_matches.size());
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(cost_time != -1) {
        std::string time_str = "FPS: " + std::to_string(1000 / cost_time);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

#endif //UTILS_H
