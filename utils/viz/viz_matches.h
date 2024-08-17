#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace slamplay {

void showKeypoints(const std::string &title, cv::Mat image, const std::vector<cv::KeyPoint> &keypoints) {
    cv::Mat image_show;
    cvtColor(image, image_show, cv::COLOR_GRAY2BGR);
    for (const cv::KeyPoint &kp : keypoints) {
        cv::circle(image_show, kp.pt, 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow(title.c_str(), image_show);
}

cv::Mat showCorrectMatches(const cv::Mat &image1, const cv::Mat &image2,
                           const std::vector<cv::KeyPoint> &keypoints1,
                           const std::vector<cv::KeyPoint> &keypoints2,
                           const std::vector<cv::DMatch> &inlierMatches,
                           const std::vector<cv::DMatch> &wrongMatches,
                           bool showSinglePoints = false) {
    cv::Mat outImage;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, wrongMatches, outImage, cv::Scalar(0, 0, 255), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(image1, keypoints1, image2, keypoints2, inlierMatches, outImage, cv::Scalar(0, 255, 0), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    if (showSinglePoints)
    {
        std::vector<bool> matched1(keypoints1.size(), true);
        std::vector<bool> matched2(keypoints2.size(), true);
        for (auto m : inlierMatches)
        {
            matched1[m.queryIdx] = false;
            matched2[m.trainIdx] = false;
        }
        for (auto m : wrongMatches)
        {
            matched1[m.queryIdx] = false;
            matched2[m.trainIdx] = false;
        }
        std::vector<cv::KeyPoint> singlePoint1, singlePoint2;
        for (size_t index = 0; index < matched1.size(); ++index)
        {
            if (matched1[index])
            {
                singlePoint1.emplace_back(keypoints1[index]);
            }
        }
        for (size_t index = 0; index < matched2.size(); ++index)
        {
            if (matched2[index])
            {
                singlePoint2.emplace_back(keypoints2[index]);
            }
        }
        cv::drawMatches(image1, singlePoint1, image2, singlePoint2, std::vector<cv::DMatch>(), outImage, cv::Scalar(0, 255, 0), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }
    return outImage;
}

inline void visualizeMatches(const cv::Mat &image0, const std::vector<cv::KeyPoint> &keypoints0,
                             const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints1,
                             const std::vector<cv::DMatch> &matches,
                             cv::Mat &output_image, double cost_time = -1,
                             const std::string &title_str = "Matches") {
    if (image0.size != image1.size) return;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, matches, output_image, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    double sc = std::min(image0.rows / 640., 2.0);
    int ht = int(30 * sc);
    cv::putText(output_image, title_str, cv::Point(int(8 * sc), ht), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, title_str, cv::Point(int(8 * sc), ht), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string feature_points_str = "Keypoints: " + std::to_string(keypoints0.size()) + ":" + std::to_string(keypoints1.size());
    cv::putText(output_image, feature_points_str, cv::Point(int(8 * sc), ht * 2), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, feature_points_str, cv::Point(int(8 * sc), ht * 2), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string match_points_str = "Matches: " + std::to_string(matches.size());
    cv::putText(output_image, match_points_str, cv::Point(int(8 * sc), ht * 3), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, match_points_str, cv::Point(int(8 * sc), ht * 3), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if (cost_time != -1) {
        std::string time_str = "FPS: " + std::to_string(1000 / cost_time);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

}  // namespace slamplay