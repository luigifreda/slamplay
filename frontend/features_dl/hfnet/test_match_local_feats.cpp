/**
 * A test for matching
 *
 *
 * Result:
HF + BFMatcher_L2:
match costs time: 12ms
matches total number: 832
threshold matches total number: 832
correct matches number: 767
match correct percentage: 0.921875
HF + BFMatcher_L1:
match costs time: 25ms
matches total number: 831
threshold matches total number: 829
correct matches number: 780
match correct percentage: 0.940893
HF + SearchByBoWV2:
match costs time: 5ms
matches total number: 934
threshold matches total number: 934
correct matches number: 745
match correct percentage: 0.797645
 * 1. HFNet is way better than ORB, but it is more time-consuming
 * 2. The L1 and L2 descriptor distance is the same for HFNet, but L2 norm is more effective
 * 3. SearchByBoW will increase the matching time
 * 4. SearchByBoW can increase the correct percentage of ORB descriptor
 * 5. SearchByBoW does not work well for HF descriptor, maybe it is because the vocabulary for HF is bad.
 * 6. The vocabulary costs too much time!
 */
#include <dirent.h>
#include <chrono>
#include <fstream>
#include <random>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "extractors/HFNetSettings.h"
#include "extractors/HFextractor.h"
#include "extractors/utility_common.h"

using namespace cv;
using namespace std;
using namespace hfnet;

int SearchByBoWHFNetSLAM(float mfNNratio, float threshold, bool mutual,
                         cv::Mat& Descriptors1, cv::Mat& Descriptors2,
                         std::vector<cv::DMatch>& vMatches) {
    vMatches.clear();
    vMatches.reserve(Descriptors1.rows);

    assert(Descriptors1.isContinuous() && Descriptors2.isContinuous());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des1(Descriptors1.ptr<float>(), Descriptors1.rows, Descriptors1.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des2(Descriptors2.ptr<float>(), Descriptors2.rows, Descriptors2.cols);
    // cv::Mat distanceCV = 2 * (1 - Descriptors1 * Descriptors2.t());
    Eigen::MatrixXf distance = 2 * (Eigen::MatrixXf::Ones(Descriptors1.rows, Descriptors2.rows) - des1 * des2.transpose());

    vector<int> matchedIdx2(Descriptors2.rows, -1);
    vector<float> matchedDist(Descriptors2.rows, std::numeric_limits<float>::max());
    for (int idx1 = 0; idx1 < distance.rows(); idx1++)
    {
        float bestDist1 = std::numeric_limits<float>::max();
        int bestIdx2 = -1;
        float bestDist2 = std::numeric_limits<float>::max();

        for (int idx2 = 0; idx2 < distance.cols(); idx2++)
        {
            float dist = distance(idx1, idx2);

            if (dist < bestDist1)
            {
                bestDist2 = bestDist1;
                bestDist1 = dist;
                bestIdx2 = idx2;
            } else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist1 < threshold)
        {
            if (bestDist1 < mfNNratio * bestDist2)
            {
                int bestCrossIdx1 = -1;
                if (mutual)
                {
                    // cross check
                    float bestDist = std::numeric_limits<float>::max();
                    for (int crossIdx1 = 0; crossIdx1 < distance.rows(); crossIdx1++)
                    {
                        float dist = distance(crossIdx1, bestIdx2);
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestCrossIdx1 = crossIdx1;
                        }
                    }
                }

                if (!mutual || bestCrossIdx1 == idx1)
                {
                    DMatch match;
                    match.queryIdx = idx1;
                    match.trainIdx = bestIdx2;
                    match.distance = bestDist1;
                    match.imgIdx = 0;
                    vMatches.emplace_back(match);
                }
            }
        }
    }

    return vMatches.size();
}

int main(int argc, char* argv[]) {
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
        std::cout << "Usage: test_match_local_feats path_to_dataset path_to_model" << std::endl;
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
    InitAllModels(strModelPath, kHFNetRTModel, ImSize, 4, 1.2f);
    // InitAllModels(strModelPath, kHFNetTFModel, ImSize, 4, 1.2f);
    auto vpModels = GetModelVec();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, files.size());

    int nFeatures = 700;
    float threshold = 0.02;
    HFextractor extractorHF(nFeatures, threshold, 1.2f, 4, vpModels);

    const cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_32F);
    char command = ' ';
    float matchThreshold = 0.6;
    float ratioThreshold = 0.9;
    bool showWholeMatches = false;
    int select = 0;
    do {
        if (command == 'w')
            select += 1;
        else if (command == 's')
            select -= 1;
        else if (command == 'a')
            threshold = std::max(threshold - 0.001, 0.005);
        else if (command == 'd')
            threshold += 0.001;
        else if (command == 'q')
            matchThreshold -= 0.05;
        else if (command == 'e')
            matchThreshold += 0.05;
        else if (command == 'r')
            ratioThreshold -= 0.1;
        else if (command == 'f')
            ratioThreshold = std::min(ratioThreshold + 0.1, 1.0);
        else if (command == 'i')
            showWholeMatches = !showWholeMatches;
        else if (command == ' ')
            select = distribution(generator);

        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "matchThreshold: " << matchThreshold << endl;
        cout << "ratioThreshold: " << ratioThreshold << endl;
        cv::Mat image1 = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat image2 = imread(strDatasetPath + files[select + 10], IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2;
        cv::Mat globalDescriptorsHF;
        extractorHF(image1, keypointsHF1, descriptorsHF1, globalDescriptorsHF);
        extractorHF(image2, keypointsHF2, descriptorsHF2, globalDescriptorsHF);

        cout << "-------------------------------------------------------" << endl;
        {
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            cv::BFMatcher cvMatcherHF(cv::NORM_L2, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            for (auto& match : matchesHF)
            {
                if (match.distance > matchThreshold) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + BFMatcher_L2", plotHF);
            cout << "HF + BFMatcher_L2:" << endl;
            cout << "match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        {
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            cv::BFMatcher cvMatcherHF(cv::NORM_L1, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            for (auto& match : matchesHF)
            {
                if (match.distance > matchThreshold * 10) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + BFMatcher_L1", plotHF);
            cout << "HF + BFMatcher_L1:" << endl;
            cout << "match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        {  // The speed is faster than BFMather, but rhe correct percentage is lower
            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, true, descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + SearchByBoWHFNetSLAM", plotHF);
            cout << "HF + SearchByBoWHFNetSLAM:" << endl;
            cout << "match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
    } while ((command = cv::waitKey()) != 'q');

    return 0;
}
