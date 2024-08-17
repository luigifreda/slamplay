#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace slamplay {

cv::Mat findCorrectMatchesByHomography(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                                       const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    if (matches.size() < 10)
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    std::vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography;
    std::vector<int> inliers;
    homography = cv::findHomography(vPt1, vPt2, cv::RANSAC, 10.0, inliers);

    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matches.size());
    wrongMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers[index])
            inlierMatches.emplace_back(matches[index]);
        else
            wrongMatches.emplace_back(matches[index]);
    }

    return homography;
}

cv::Mat findCorrectMatchesByEssentialMat(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches,
                                         const cv::Mat &cameraMatrix, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    if (matches.size() < 10)
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    std::vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat E = findEssentialMat(vPt1, vPt2, cameraMatrix, cv::RANSAC, 0.999, 3.0, inliers);
    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matches.size());
    wrongMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers.at<uchar>(index))
            inlierMatches.emplace_back(matches[index]);
        else
            wrongMatches.emplace_back(matches[index]);
    }

    return E;
}

cv::Mat findCorrectMatchesByPnP(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &depthImage1, const std::vector<cv::KeyPoint> &keypoints2,
                                const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                std::vector<cv::DMatch> &matches,
                                std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    std::vector<cv::Point3f> vPt1;
    std::vector<cv::Point2f> vPt2;
    std::vector<cv::DMatch> matchesTemp;
    for (const auto &match : matches)
    {
        auto kpt1 = keypoints1[match.queryIdx].pt;
        float depth1 = depthImage1.ptr<unsigned short>(int(kpt1.y))[int(kpt1.x)] / 5000.0;
        if (depth1 == 0) continue;

        float ptX = (kpt1.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
        float ptY = (kpt1.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(1, 1);
        vPt1.push_back(cv::Point3f(ptX * depth1, ptY * depth1, depth1));

        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
        matchesTemp.emplace_back(match);
    }

    matches = matchesTemp;
    if (vPt1.size() < 10)
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    solvePnPRansac(vPt1, vPt2, cameraMatrix, distCoeffs, rvec, tvec, false, 300, 8.0f, 0.99, inliers, cv::SOLVEPNP_EPNP);
    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matchesTemp.size());
    wrongMatches.reserve(matchesTemp.size());
    int inlier = 0, index = 0;
    while (index < matchesTemp.size())
    {
        if (inlier < inliers.rows && index == inliers.at<int>(inlier))
        {
            inlierMatches.emplace_back(matchesTemp[index++]);
            inlier++;
        } else
        {
            wrongMatches.emplace_back(matchesTemp[index++]);
        }
    }
    return cv::Mat();
}

}  // namespace slamplay