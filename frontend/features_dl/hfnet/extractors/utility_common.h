#ifndef UTILITY_COMMON_H
#define UTILITY_COMMON_H

#include <dirent.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "extractors/BaseModel.h"

using namespace cv;
using namespace std;

struct TicToc {
    TicToc() {};
    void clearBuff() { timeBuff.clear(); }
    void Tic() { t1 = chrono::steady_clock::now(); }
    float Toc() {
        t2 = chrono::steady_clock::now();
        float time = chrono::duration<float, std::milli>(t2 - t1).count();
        timeBuff.emplace_back(time);
        return time;
    }
    float aveCost(void) {
        if (timeBuff.empty()) return 0;
        return std::accumulate(timeBuff.begin(), timeBuff.end(), 0.f) / (float)timeBuff.size();
    }
    float devCost(void) {
        if (timeBuff.size() <= 1) return 0;
        float average = aveCost();

        float accum = 0;
        int total = 0;
        for (double value : timeBuff)
        {
            if (value == 0)
                continue;
            accum += pow(value - average, 2);
            total++;
        }
        return sqrt(accum / total);
    }
    bool empty() { return timeBuff.empty(); }

    std::vector<float> timeBuff;
    chrono::steady_clock::time_point t1;
    chrono::steady_clock::time_point t2;
};

vector<string> GetPngFiles(string strPngDir) {
    std::cout << "Reding png files from " << strPngDir << std::endl;
    struct dirent **namelist;
    std::vector<std::string> ret;
    int n = scandir(strPngDir.c_str(), &namelist, [](const struct dirent *cur) -> int {
        std::string str(cur->d_name);
        return str.find(".png") != std::string::npos; }, alphasort);

    if (n < 0) {
        return ret;
    }

    for (int i = 0; i < n; i++) {
        std::string filepath(namelist[i]->d_name);
        ret.push_back("/" + filepath);
    }

    free(namelist);
    return ret;
}

void ShowKeypoints(const string &title, Mat image, const std::vector<KeyPoint> &keypoints) {
    Mat image_show;
    cvtColor(image, image_show, COLOR_GRAY2BGR);

    for (const KeyPoint &kp : keypoints) {
        cv::circle(image_show, kp.pt, 2, Scalar(0, 255, 0), -1);
    }

    cv::imshow(title.c_str(), image_show);
}

cv::Mat FindCorrectMatchesByHomography(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                                       const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    if (matches.size() < 10)
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    vector<cv::Point2f> vPt1, vPt2;
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

cv::Mat FindCorrectMatchesByEssentialMat(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches,
                                         const cv::Mat &cameraMatrix, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    if (matches.size() < 10)
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    vector<cv::Point2f> vPt1, vPt2;
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

cv::Mat FindCorrectMatchesByPnP(const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &depthImage1, const std::vector<cv::KeyPoint> &keypoints2,
                                const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                std::vector<cv::DMatch> &matches,
                                std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches) {
    vector<cv::Point3f> vPt1;
    vector<cv::Point2f> vPt2;
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

cv::Mat ShowCorrectMatches(const cv::Mat &image1, const cv::Mat &image2,
                           const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                           const std::vector<cv::DMatch> &inlierMatches, const std::vector<cv::DMatch> &wrongMatches, bool showSinglePoints = false) {
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

std::vector<cv::KeyPoint> undistortPoints(const std::vector<cv::KeyPoint> &src, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    std::vector<cv::Point2f> mat;
    std::vector<cv::KeyPoint> res;
    for (auto temp : src)
    {
        mat.emplace_back(temp.pt);
    }
    cv::undistortPoints(mat, mat, cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix);
    for (int index = 0; index < mat.size(); ++index)
    {
        auto kpt = src[index];
        kpt.pt = mat[index];
        res.emplace_back(kpt);
    }
    return res;
}

#endif