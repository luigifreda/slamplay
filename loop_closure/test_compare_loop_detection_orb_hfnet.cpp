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
#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "features/ORBVocabulary.h"
#include "features/ORBextractor.h"
#if USE_TENSORFLOW || USE_TENSORRT
#include "features_dl/hfnet/HFNetSettings.h"
#include "features_dl/hfnet/HFextractor.h"
#endif

#include <cv/matches_utils.h>
#include <io/file_utils.h>
#include <io/image_io.h>
#include <time/TicToc.h>

#include "io/messages.h"
#include "macros.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace slamplay;
#if USE_TENSORFLOW || USE_TENSORRT
using namespace hfnet;
#endif

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

struct TestKeyFrame {
    int mnFrameId;
    float mPlaceRecognitionScore = 1.0;
};

#if USE_TENSORFLOW || USE_TENSORRT
struct KeyFrameHFNetSLAM : public TestKeyFrame {
    cv::Mat mGlobalDescriptors;

    KeyFrameHFNetSLAM(const int id, const cv::Mat im, HFNetBaseModel *pModel) {
        mnFrameId = id;
        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, intermediate;
        pModel->Detect(im, vKeyPoints, localDescriptors, mGlobalDescriptors, 1000, 0.01);
    }
};
#endif

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors) {
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

struct KeyFrameORBSLAM3 : public TestKeyFrame {
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    KeyFrameORBSLAM3(const int id, const cv::Mat im, ORBVocabulary *mpORBvocabulary, ORBextractor *extractorLeft) {
        mnFrameId = id;
        vector<int> vLapping = {0, 1000};
        (*extractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
};

typedef vector<TestKeyFrame *> KeyFrameDB;

KeyFrameDB GetNCandidateLoopFrameORBSLAM3(ORBVocabulary *mpVoc, KeyFrameORBSLAM3 *query, const KeyFrameDB &db, int k) {
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameORBSLAM3 *pKF = static_cast<KeyFrameORBSLAM3 *>(*it);
        if (pKF->mnFrameId >= query->mnFrameId - 30) break;
        count++;
        pKF->mPlaceRecognitionScore = mpVoc->score(pKF->mBowVec, query->mBowVec);
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](TestKeyFrame *const f1, TestKeyFrame *const f2) {
        return f1->mPlaceRecognitionScore > f2->mPlaceRecognitionScore;
    });
    return res;
}

#if USE_TENSORFLOW || USE_TENSORRT
KeyFrameDB GetNCandidateLoopFrameHFNetSLAM(KeyFrameHFNetSLAM *query, const KeyFrameDB &db, int k) {
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, query->mGlobalDescriptors.cols);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = static_cast<KeyFrameHFNetSLAM *>(*it);
        if (pKF->mnFrameId >= query->mnFrameId - 30) break;
        count++;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pKFDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = 1 - (queryDescriptors - pKFDescriptors).norm();
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](TestKeyFrame *const f1, TestKeyFrame *const f2) {
        return f1->mPlaceRecognitionScore > f2->mPlaceRecognitionScore;
    });
    return res;
}
#endif

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str) {
    cv::Mat plot;
    cv::cvtColor(image, plot, cv::COLOR_GRAY2RGB);
    cv::putText(plot, str, cv::Point2d(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::imshow(title, plot);
}

int main(int argc, char **argv) {
    std::string strDatasetPath;
    std::string strModelPath;
    std::string strVocFileORB;
    int nLevels = 4;
    float scaleFactor = 1.2f;

#if USE_TENSORFLOW || USE_TENSORRT
    hfnet::HFNetSettings settings;
#endif

    if (argc == 4) {
        strDatasetPath = std::string(argv[1]);
        strModelPath = std::string(argv[2]);
        strVocFileORB = std::string(argv[3]);
    } else
    {
        std::cout << "Usage: compare_loop_detection path_to_dataset path_to_model path_to_vocabulary" << endl;
#if USE_TENSORFLOW || USE_TENSORRT
        strDatasetPath = settings.strDatasetPath();
        strModelPath = settings.strModelPath();
        nLevels = settings.nLevels();
        scaleFactor = settings.scaleFactor();
#endif
        strVocFileORB = dataDir + "/dbow2_vocabulary/ORBvoc.bin";
    }

    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

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
    std::cout << "Got [" << files.size() << "] images in dataset" << std::endl;

#if USE_TENSORFLOW || USE_TENSORRT
    HFNetBaseModel *pModel = InitRTModel(strModelPath, kImageToLocalAndGlobal, {1, ImSize.height, ImSize.width, 1});
    // HFNetBaseModel *pModel = InitTFModel(strModelPath, kImageToLocalAndGlobal, {1, ImSize.height, ImSize.width, 1});
#endif

    ORBextractor extractorORB(1000, 1.2, 8, 20, 7);

    ORBVocabulary vocabORB;
    if (hasFileSuffix(strVocFileORB, ".bin")) {
        if (!vocabORB.loadFromBinaryFile(strVocFileORB))
        {
            cerr << "Falied to open at: " << strVocFileORB << endl;
            exit(-1);
        }
    } else {
        if (!vocabORB.loadFromTextFile(strVocFileORB))
        {
            cerr << "Falied to open at: " << strVocFileORB << endl;
            exit(-1);
        }
    }

    int start = 0;
    int end = files.size();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(30, end);

    const int step = 4;
    int nKeyFrame = (end - start) / step;

    // if (nKeyFrame <= 30) exit(-1);
    std::cout << "Dataset range: [" << start << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << std::endl;

    std::cout << "Loading dataset..." << std::endl;
    KeyFrameDB vKeyFrameDBHFNetSLAM, vKeyFrameDBORBSLAM3;
    vKeyFrameDBHFNetSLAM.reserve(nKeyFrame);
    vKeyFrameDBORBSLAM3.reserve(nKeyFrame);
    float cur = start;
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

#if USE_TENSORFLOW || USE_TENSORRT
        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModel);
        vKeyFrameDBHFNetSLAM.emplace_back(pKFHF);
#endif

        KeyFrameORBSLAM3 *pKFORB = new KeyFrameORBSLAM3(select, image, &vocabORB, &extractorORB);
        vKeyFrameDBORBSLAM3.emplace_back(pKFORB);

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

    char command = 0;
    int select = 0;
    while (select < files.size())
    {
        bool show_key_interation = true;
        if (command == 'w') {
            select += 1;
        } else if (command == 'x') {
            select -= 1;
        } else if (command == ' ') {
            select = distribution(generator);
        } else {
            select++;
            show_key_interation = false;
        }

        const size_t selectChecked = std::clamp((size_t)select, (size_t)0, files.size() - 1);

        cv::Mat image = imread(strDatasetPath + files[selectChecked], IMREAD_GRAYSCALE);

#if USE_TENSORFLOW || USE_TENSORRT
        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(selectChecked, image, pModel);
#endif

        KeyFrameORBSLAM3 *pKFORB = new KeyFrameORBSLAM3(selectChecked, image, &vocabORB, &extractorORB);

        vector<TestKeyFrame *> resOrb;
        vector<TestKeyFrame *> resHfnet;

#if USE_TENSORFLOW || USE_TENSORRT
        auto t1_H = chrono::steady_clock::now();
        resHfnet = GetNCandidateLoopFrameHFNetSLAM(pKFHF, vKeyFrameDBHFNetSLAM, 3);
        auto t2_H = chrono::steady_clock::now();
        auto tH = chrono::duration_cast<chrono::microseconds>(t2_H - t1_H).count();
#endif

        auto t1_O = chrono::steady_clock::now();
        resOrb = GetNCandidateLoopFrameORBSLAM3(&vocabORB, pKFORB, vKeyFrameDBORBSLAM3, 3);
        auto t2_O = chrono::steady_clock::now();
        auto tO = chrono::duration_cast<chrono::microseconds>(t2_O - t1_O).count();

#if USE_TENSORFLOW || USE_TENSORRT
        std::cout << "HFNet-SLAM: " << std::endl;
        std::cout << "\t Query cost time: " << tH << std::endl;
#endif

        std::cout << "ORB-SLAM3: " << std::endl;
        std::cout << "\t Query cost time: " << tO << std::endl;

        ShowImageWithText("Query Image", image, std::to_string((int)pKFORB->mnFrameId));
        cv::moveWindow("Query Image", 0, 0);

        auto showImages = [&](vector<TestKeyFrame *> &res, const std::string &titlePrefix, const int rowShow = 0) {
            const int deltaH = 100;
            const int deltaV = 100;
            for (size_t index = 0; index < 3; ++index) {
                std::string windowTile = titlePrefix + " - Candidate " + std::to_string(index + 1);
                if (index < res.size()) {
                    const size_t idxChecked = std::clamp((size_t)res[index]->mnFrameId, (size_t)0, files.size() - 1);
                    cv::Mat image = imread(strDatasetPath + files[idxChecked], IMREAD_GRAYSCALE);
                    ShowImageWithText(windowTile, image,
                                      "F:" + std::to_string((int)res[index]->mnFrameId) + ", d:" + std::to_string(res[index]->mPlaceRecognitionScore));
                } else {
                    Mat empty = cv::Mat::zeros(ImSize, CV_8U);
                    cv::imshow(windowTile, empty);
                }
                cv::moveWindow(windowTile, image.cols * (index + 1) + deltaH, rowShow * (image.rows + deltaV));
            }
        };

#if USE_TENSORFLOW || USE_TENSORRT
        showImages(resHfnet, "HFNet", 0);
#endif

        showImages(resOrb, "ORB", 1);

        command = cv::waitKey();

        std::cout << "===========================================================" << std::endl;
    }

    system("pause");
}