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
/**
 *
GetNCandidateLoopFrameCV():
Query cost time: 1339
Query cost time: 1328
GetNCandidateLoopFrameEigen():
Query cost time: 245
Query cost time: 259
 * Eigen is much faster than OpenCV
 */
#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include <cv/matches_utils.h>
#include <io/image_io.h>
#include "features_dl/hfnet/HFNetSettings.h"
#include "features_dl/hfnet/HFextractor.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace hfnet;
using namespace slamplay;

struct KeyFrameHFNetSLAM {
    cv::Mat mGlobalDescriptors;
    int mnFrameId;
    float mPlaceRecognitionScore = 1.0;

    KeyFrameHFNetSLAM(int id, const cv::Mat im, HFNetBaseModel *pModel) {
        mnFrameId = id;
        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, intermediate;
        pModel->Detect(im, vKeyPoints, localDescriptors, mGlobalDescriptors, 1000, 0.01);
    }
};

typedef vector<KeyFrameHFNetSLAM *> KeyFrameDB;

KeyFrameDB GetNCandidateLoopFrameCV(KeyFrameHFNetSLAM *query, const KeyFrameDB &db, int k) {
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 30) break;
        count++;
        pKF->mPlaceRecognitionScore = cv::norm(query->mGlobalDescriptors - pKF->mGlobalDescriptors, cv::NORM_L2);
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](KeyFrameHFNetSLAM *const f1, KeyFrameHFNetSLAM *const f2) {
        return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
    });
    return res;
}

KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameHFNetSLAM *query, const KeyFrameDB &db, int k) {
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, query->mGlobalDescriptors.cols);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 30) break;
        count++;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pKFDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = (queryDescriptors - pKFDescriptors).norm();
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](KeyFrameHFNetSLAM *const f1, KeyFrameHFNetSLAM *const f2) {
        return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
    });
    return res;
}

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str) {
    cv::Mat plot;
    cv::cvtColor(image, plot, cv::COLOR_GRAY2RGB);
    cv::putText(plot, str, cv::Point2d(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::imshow(title, plot);
}

int main(int argc, char **argv) {
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    std::string strDatasetPath;
    std::string strModelPath;
    int nLevels = 4;
    float scaleFactor = 1.2;

    hfnet::HFNetSettings settings;

    if (argc == 3) {
        strDatasetPath = argv[1];
        strModelPath = argv[2];

    } else
    {
        std::cout << "Usage: test_match_global_feats path_to_dataset path_to_model" << std::endl;
        strDatasetPath = settings.strDatasetPath();
        strModelPath = settings.strModelPath();
        nLevels = settings.nLevels();
        scaleFactor = settings.scaleFactor();
    }

    vector<string> files = GetPngFiles(strDatasetPath);  // get all image files
    if (files.empty()) {
        std::cout << "Error, failed to find any valid image in: " << strDatasetPath << std::endl;
        return 1;
    }
    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        std::cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << std::endl;
        return 1;
    }

    cv::Vec4i inputShape{1, ImSize.height, ImSize.width, 1};
    auto pModel = InitRTModel(strModelPath, kImageToLocalAndGlobal, inputShape);

    int start = 0;
    int end = files.size();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(30, end);

    const int step = 4;
    int nKeyFrame = (end - start) / step;

    // if (nKeyFrame <= 30) exit(-1);
    std::cout << "Dataset range: [" << start << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << std::endl;

    std::cout << "Loading dataset..." << std::endl;
    KeyFrameDB vKeyFrameDB;
    vKeyFrameDB.reserve(nKeyFrame);
    float cur = start;
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, globalDescriptors;

        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModel);
        vKeyFrameDB.emplace_back(pKFHF);
        cur += step;
    }

    // cv::namedWindow("Query Image");
    // cv::moveWindow("Query Image", 0, 0);
    // cv::namedWindow("Candidate 1");
    // cv::moveWindow("Candidate 1", 820, 0);
    // cv::namedWindow("Candidate 2");
    // cv::moveWindow("Candidate 2", 0, 540);
    // cv::namedWindow("Candidate 3");
    // cv::moveWindow("Candidate 3", 820, 540);

    char command = ' ';
    int select = 0;
    while (select < files.size() && command != 'q')
    {
        if (command == 'w')
            select += 1;
        else if (command == 'x')
            select -= 1;
        else if (command == 'r')
            select = distribution(generator);
        else {
            select++;
        }

        cout << "====================================================" << endl;

        cv::Mat image = cv::imread(strDatasetPath + files[select], cv::IMREAD_GRAYSCALE);

        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModel);

        auto t1 = chrono::steady_clock::now();
        auto res = GetNCandidateLoopFrameEigen(pKFHF, vKeyFrameDB, 3);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        // std::cout << "Found " << res.size() << " candidate loop frames" << std::endl;
        std::cout << "Query cost time: " << t << " ms" << std::endl;

        ShowImageWithText("Query Image", image, std::to_string((int)pKFHF->mnFrameId));

        auto showImages = [&](KeyFrameDB &res, const std::string &titlePrefix, const int rowShow = 0) {
            const int deltaH = 100;
            const int deltaV = 100;
            for (size_t index = 0; index < 3; ++index) {
                std::string windowTile = titlePrefix + " - Candidate " + std::to_string(index + 1);
                if (index < res.size()) {
                    const size_t idxChecked = std::clamp((size_t)res[index]->mnFrameId, (size_t)0, files.size() - 1);
                    cv::Mat image = cv::imread(strDatasetPath + files[idxChecked], cv::IMREAD_GRAYSCALE);
                    ShowImageWithText(windowTile, image,
                                      "F:" + std::to_string((int)res[index]->mnFrameId) + ", d:" + std::to_string(res[index]->mPlaceRecognitionScore));
                } else {
                    Mat empty = cv::Mat::zeros(ImSize, CV_8U);
                    cv::imshow(windowTile, empty);
                }
                cv::moveWindow(windowTile, image.cols * (index + 1) + deltaH, rowShow * (image.rows + deltaV));
            }
        };

        showImages(res, "HFNet", 0);

        command = cv::waitKey();
    }

    return 0;
}