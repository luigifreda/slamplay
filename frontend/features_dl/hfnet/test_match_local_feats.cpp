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

#include <cv/matches_utils.h>
#include <io/image_io.h>
#include <time/TicToc.h>
#include "features_dl/hfnet/HFNetSettings.h"
#include "features_dl/hfnet/HFextractor.h"

#include "viz/viz_matches.h"

using namespace cv;
using namespace std;
using namespace hfnet;
using namespace slamplay;

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
    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
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

    // cv::Vec4i inputShape{1, ImSize.height, ImSize.width, 1};

    const ModelType modelType = kHFNetRTModel;
    // const ModelType modelType = kHFNetTFModel;  // only when tensorflow is available and USE_TENSORFLOW is defined
    InitAllModels(strModelPath, kHFNetRTModel, ImSize, nLevels, scaleFactor);

    auto vpModels = GetModelVec();

    std::default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> distribution(0, files.size() - 20);

    int nFeatures = 700;
    float threshold = 0.02;
    HFextractor extractorHF(nFeatures, threshold, scaleFactor, nLevels, vpModels);

    const cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_32F);
    char command = ' ';
    float matchThreshold = 0.6;
    float ratioThreshold = 0.9;
    bool showWholeMatches = false;
    const int deltaSelect = 10;
    int select = 0;
    do {
        bool show_key_interation = false;
        if (command == 'x') {
            break;
        } else if (command == 'w') {
            show_key_interation = true;
            select += 1;
        } else if (command == 's') {
            show_key_interation = true;
            select -= 1;
        } else if (command == 'a') {
            show_key_interation = true;
            threshold = std::max(threshold - 0.001, 0.005);
        } else if (command == 'd') {
            show_key_interation = true;
            threshold += 0.001;
        } else if (command == 'q') {
            show_key_interation = true;
            matchThreshold -= 0.05;
        } else if (command == 'e') {
            show_key_interation = true;
            matchThreshold += 0.05;
        } else if (command == 'r') {
            show_key_interation = true;
            ratioThreshold -= 0.1;
        } else if (command == 'f') {
            show_key_interation = true;
            ratioThreshold = std::min(ratioThreshold + 0.1, 1.0);
        } else if (command == 'i') {
            show_key_interation = true;
            showWholeMatches = !showWholeMatches;
        } else if (command == 'g') {
            show_key_interation = true;
            select = distribution(generator);
        } else {
            select++;
        }
        if (show_key_interation) {
            cout << "command: " << command << endl;
            cout << "select: " << select << endl;
            cout << "threshold: " << threshold << endl;
            cout << "matchThreshold: " << matchThreshold << endl;
            cout << "ratioThreshold: " << ratioThreshold << endl;
        }

        if (select >= files.size() - deltaSelect) break;

        size_t select1 = std::clamp((size_t)select, (size_t)0, files.size() - 1);
        size_t select2 = std::clamp((size_t)select + deltaSelect, (size_t)0, files.size() - 1);
        // std::cout << "reading: " << files[select1] << " and " << files[select2] << std::endl;

        cv::Mat image1 = cv::imread(strDatasetPath + files[select1], cv::IMREAD_GRAYSCALE);
        cv::Mat image2 = cv::imread(strDatasetPath + files[select2], cv::IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2, globalDescriptorsHF;
        extractorHF(image1, keypointsHF1, descriptorsHF1, globalDescriptorsHF);
        extractorHF(image2, keypointsHF2, descriptorsHF2, globalDescriptorsHF);

        cout << "====================================================" << endl;
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
            cv::Mat E = findCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = showCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + BFMatcher_L2", plotHF);
            cout << "HF + BFMatcher_L2:" << endl;
            cout << "\t match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "\t matches total number: " << matchesHF.size() << endl;
            cout << "\t threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "\t correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "\t match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
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
            cv::Mat E = findCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = slamplay::showCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + BFMatcher_L1", plotHF);
            cout << "HF + BFMatcher_L1:" << endl;
            cout << "\t match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "\t matches total number: " << matchesHF.size() << endl;
            cout << "\t threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "\t correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "\t match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        {  // The speed is faster than BFMather, but rhe correct percentage is lower
            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, true, descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            cv::Mat E = findCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cv::Mat::eye(3, 3, CV_32F), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = slamplay::showCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + SearchByBoWHFNetSLAM", plotHF);
            cout << "HF + SearchByBoWHFNetSLAM:" << endl;
            cout << "\t match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "\t matches total number: " << matchesHF.size() << endl;
            cout << "\t correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "\t match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
    } while ((command = cv::waitKey()) != 'q');

    return 0;
}
