/**
 * To Test the performance of different dector
 *
================= HFNet TensorRT Float 32: kImageToLocalAndGlobal =====================
Evaluate the run time perfomance in dataset:

Detect the full features with TestExtractor ExtractUsingParallel():
Get features number: 1000
pyramid costs: 0.370841 ± 0.202294
level 0 costs : 11.4856 ± 0.569528
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0.0685001 ± 0.0113844
total costs: 11.9271 ± 0.685793

Detect the full features with TestExtractor ExtractUsingFor():
Get features number: 1000
pyramid costs: 0.356541 ± 0.141287
level 0 costs : 5.95567 ± 0.33487
level 1 costs : 3.09855 ± 0.199504
level 2 costs : 2.37011 ± 0.136594
level 3 costs : 1.73447 ± 0.101819
copy costs: 0.0695352 ± 0.0169045
total costs: 13.587 ± 0.889493

Detect the full features with HFextractor:
Get features number: 1000
pyramid costs: 0 ± 0
level 0 costs : 0 ± 0
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0 ± 0
total costs: 12.1665 ± 0.829897

================= HFNet TensorRT Float 16: kImageToLocalAndGlobal =====================
Evaluate the run time perfomance in dataset:

Detect the full features with TestExtractor ExtractUsingParallel():
Get features number: 1000
pyramid costs: 0.314499 ± 0.030389
level 0 costs : 7.14264 ± 0.23482
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0.0684819 ± 0.00474888
total costs: 7.52749 ± 0.240746

Detect the full features with TestExtractor ExtractUsingFor():
Get features number: 1000
pyramid costs: 0.323611 ± 0.0215451
level 0 costs : 3.67066 ± 0.0755506
level 1 costs : 2.105 ± 0.060039
level 2 costs : 1.54658 ± 0.0457351
level 3 costs : 1.18301 ± 0.0288683
copy costs: 0.0645909 ± 0.00581761
total costs: 8.89519 ± 0.181361

Detect the full features with HFextractor:
Get features number: 1000
pyramid costs: 0 ± 0
level 0 costs : 0 ± 0
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0 ± 0
total costs: 7.56352 ± 0.117819

================= HFNet TensorFlow: kImageToLocalAndIntermediate =====================
Evaluate the run time perfomance in dataset:

Detect the full features with TestExtractor ExtractUsingParallel():
Get features number: 1000
pyramid costs: 0.290319 ± 0.0336965
level 0 costs : 16.6114 ± 0.745647
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0.0771891 ± 0.0142136
total costs: 16.9817 ± 0.757257

Detect the full features with TestExtractor ExtractUsingFor():
Get features number: 1000
pyramid costs: 0.315143 ± 0.028732
level 0 costs : 7.36702 ± 0.483607
level 1 costs : 5.22422 ± 0.324976
level 2 costs : 4.00371 ± 0.340359
level 3 costs : 3.02608 ± 0.225555
copy costs: 0.0863364 ± 0.0205967
total costs: 20.0252 ± 0.986224

Detect the full features with HFextractor:
Get features number: 1000
pyramid costs: 0 ± 0
level 0 costs : 0 ± 0
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
copy costs: 0 ± 0
total costs: 17.4885 ± 1.36514


================= HFNet VINO: kImageToLocalAndIntermediate =====================
Evaluate the run time perfomance in dataset:

Detect the full features with TestExtractor ExtractUsingParallel():
Get features number: 728
pyramid costs: 1.30138 ± 1.69273
level 0 costs : 170.373 ± 42.9945
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
level 4 costs : 0 ± 0
level 5 costs : 0 ± 0
level 6 costs : 0 ± 0
level 7 costs : 0 ± 0
copy costs: 0.104862 ± 0.0486709
total costs: 171.792 ± 43.0424

Detect the full features with TestExtractor ExtractUsingFor():
Get features number: 1123
pyramid costs: 2.41067 ± 1.14057
level 0 costs : 28.6432 ± 0.660343
level 1 costs : 9.86764 ± 0.401312
level 2 costs : 6.57248 ± 0.245541
level 3 costs : 4.60928 ± 0.149062
level 4 costs : 3.01519 ± 0.108606
level 5 costs : 2.18284 ± 0.0931418
level 6 costs : 1.62319 ± 0.0720224
level 7 costs : 1.1764 ± 0.0617994
copy costs: 0.174835 ± 0.113503
total costs: 60.2931 ± 1.74302

Detect the full features with HFextractor:
Get features number: 728
pyramid costs: 0 ± 0
level 0 costs : 0 ± 0
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
level 4 costs : 0 ± 0
level 5 costs : 0 ± 0
level 6 costs : 0 ± 0
level 7 costs : 0 ± 0
copy costs: 0 ± 0
total costs: 60.0883 ± 4.10698
 */
#include <dirent.h>
#include <chrono>
#include <fstream>
#include <random>

#include <Eigen/Core>

#include <cv/matches_utils.h>
#include <io/image_io.h>
#include <time/TicToc.h>
#include "extractors/HFNetSettings.h"
#include "extractors/HFextractor.h"

#include "macros.h"
#include "messages.h"

using namespace cv;
using namespace std;
using namespace hfnet;

const int maxNumLevels = 20;

TicToc TimerPyramid;
TicToc TimerDetectPerLevel[maxNumLevels];
TicToc TimerCopy;
TicToc TimerTotal;

void ClearTimer() {
    TimerPyramid.clearBuff();
    for (auto &timer : TimerDetectPerLevel) timer.clearBuff();
    TimerCopy.clearBuff();
    TimerTotal.clearBuff();
}

void PrintTimer(int nLevels) {
    cout << "pyramid costs: " << TimerPyramid.aveCost() << " ± " << TimerPyramid.devCost() << endl;
    for (int level = 0; level < nLevels; ++level)
        cout << "level " << level << " costs : " << TimerDetectPerLevel[level].aveCost() << " ± " << TimerDetectPerLevel[level].devCost() << endl;
    cout << "copy costs: " << TimerCopy.aveCost() << " ± " << TimerCopy.devCost() << endl;
    cout << "total costs: " << TimerTotal.aveCost() << " ± " << TimerTotal.devCost() << endl;
}

struct TestExtractor : public HFextractor {
    TestExtractor(int nfeatures, float threshold, float scaleFactor,
                  int nlevels, const std::vector<BaseModel *> &vpModels) : HFextractor(nfeatures, threshold, scaleFactor, nlevels, vpModels) {}

    int ExtractUsingFor(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                        cv::Mat &localDescriptors, cv::Mat &globalDescriptors) {
        if (image.empty()) return -1;
        assert(image.type() == CV_8UC1);

        TimerPyramid.Tic();
        ComputePyramid(image);
        TimerPyramid.Toc();

        int nKeypoints = 0;
        vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
        vector<cv::Mat> allDescriptors(nlevels);
        for (int level = 0; level < nlevels; ++level)
        {
            TimerDetectPerLevel[level].Tic();
            if (level == 0)
            {
                mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], globalDescriptors, mnFeaturesPerLevel[level], threshold);
            } else
            {
                mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], mnFeaturesPerLevel[level], threshold);
            }
            TimerDetectPerLevel[level].Toc();
            nKeypoints += allKeypoints[level].size();
            // showKeypoints(std::to_string(level), mvImagePyramid[level], allKeypoints[level]);
        }

        TimerCopy.Tic();
        vKeyPoints.clear();
        vKeyPoints.reserve(nKeypoints);
        for (int level = 0; level < nlevels; ++level)
        {
            for (auto keypoint : allKeypoints[level])
            {
                keypoint.octave = level;
                keypoint.pt *= mvScaleFactor[level];
                vKeyPoints.emplace_back(keypoint);
            }
        }
        cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);
        TimerCopy.Toc();

        return vKeyPoints.size();
    }

    class DetectParallel : public cv::ParallelLoopBody {
       public:
        DetectParallel(vector<cv::KeyPoint> *allKeypoints, cv::Mat *allDescriptors, cv::Mat *globalDescriptors, TestExtractor *pExtractor)
            : mAllKeypoints(allKeypoints), mAllDescriptors(allDescriptors), mGlobalDescriptors(globalDescriptors), mpExtractor(pExtractor) {}

        virtual void operator()(const cv::Range &range) const CV_OVERRIDE {
            for (int level = range.start; level != range.end; ++level)
            {
                if (level == 0)
                {
                    mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], *mGlobalDescriptors, mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold);
                } else
                {
                    mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold);
                }
            }
        }

        DetectParallel &operator=(const DetectParallel &) {
            return *this;
        };

       private:
        vector<cv::KeyPoint> *mAllKeypoints;
        cv::Mat *mAllDescriptors;
        cv::Mat *mGlobalDescriptors;
        TestExtractor *mpExtractor;
    };

    int ExtractUsingParallel(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                             cv::Mat &localDescriptors, cv::Mat &globalDescriptors, const int nLevels) {
        if (image.empty()) return -1;
        assert(image.type() == CV_8UC1);

        TimerPyramid.Tic();
        ComputePyramid(image);
        TimerPyramid.Toc();

        int nKeypoints = 0;
        vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
        vector<cv::Mat> allDescriptors(nlevels);

        TimerDetectPerLevel[0].Tic();
        DetectParallel detector(allKeypoints.data(), allDescriptors.data(), &globalDescriptors, this);
        cv::parallel_for_(cv::Range(0, nLevels), detector);
        TimerDetectPerLevel[0].Toc();

        for (int level = 0; level < nlevels; ++level)
            nKeypoints += allKeypoints[level].size();

        TimerCopy.Tic();
        vKeyPoints.clear();
        vKeyPoints.reserve(nKeypoints);
        for (int level = 0; level < nlevels; ++level)
        {
            for (auto keypoint : allKeypoints[level])
            {
                keypoint.octave = level;
                keypoint.pt *= mvScaleFactor[level];
                vKeyPoints.emplace_back(keypoint);
            }
        }
        cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);
        TimerCopy.Toc();

        return vKeyPoints.size();
    }
};

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
        std::cout << "Usage: test_extractors path_to_dataset path_to_model" << std::endl;
        strDatasetPath = settings.strDatasetPath();
        strModelPath = settings.strModelPath();
        nLevels = settings.nLevels();
        scaleFactor = settings.scaleFactor();
    }

    MSG_ASSERT(nLevels <= maxNumLevels, "The number of levels should be less than: " << maxNumLevels);

    std::vector<std::string> files = GetPngFiles(strDatasetPath);  // get all image files
    if (files.empty()) {
        cout << "Error, failed to find any valid image in: " << strDatasetPath << endl;
        return 1;
    } else
    {
        cout << "Found " << files.size() << " images in: " << strDatasetPath << endl;
    }

    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << endl;
        return 1;
    }

#if 0
    InitAllModels(strModelPath, kHFNetRTModel, ImSize, nLevels, scaleFactor);
#else
    InitAllModels(strModelPath, kHFNetTFModel, ImSize, nLevels, scaleFactor);
#endif
    auto vpModels = GetModelVec();

    TestExtractor *pExtractor = new TestExtractor(1000, 0.01, scaleFactor, nLevels, vpModels);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, files.size());

    cv::Mat image;
    std::vector<cv::KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, globalDescriptors;

    // randomly detect an image and show the results
    char command = ' ';
    float threshold = 0.005;
    int select = 0;
    while (select < files.size())
    {
        if (command == 'q')
            break;
#if 0            
        else if (command == 's')
            select = std::max(select - 1, 0);
        else if (command == 'w')
            select += 1;
        else if (command == 'a')
            threshold = std::max(threshold - 0.005, 0.005);
        else if (command == 'd')
            threshold += 0.005;
        else
            select = distribution(generator);

        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
#endif

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        ClearTimer();
        TimerTotal.Tic();
        pExtractor->ExtractUsingParallel(image, vKeyPoints, localDescriptors, globalDescriptors, nLevels);
        TimerTotal.Toc();

        cout << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer(nLevels);

        showKeypoints("press 'q' to exit", image, vKeyPoints);
        cout << endl;
        command = cv::waitKey(1);
        select++;
    }
    cv::destroyAllWindows();

#define RUN_PERFORMANCE_TEST 0
#if RUN_PERFORMANCE_TEST
    cout << "======================================" << endl
         << "Evaluate the run time perfomance in dataset: " << endl;

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            TimerTotal.Tic();
            pExtractor->ExtractUsingParallel(image, vKeyPoints, localDescriptors, globalDescriptors, nLevels);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with TestExtractor ExtractUsingParallel(): " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer(nLevels);
    }

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            TimerTotal.Tic();
            pExtractor->ExtractUsingFor(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with TestExtractor ExtractUsingFor(): " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer(nLevels);
    }

    {
        cout << endl;
        ClearTimer();

        HFextractor extractor = HFextractor(1000, 0.01, scaleFactor, nLevels, vpModels);
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            TimerTotal.Tic();
            extractor(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with HFextractor: " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer(nLevels);
    }
#endif

    cout << endl
         << "Press 'ENTER' to exit" << endl;
    getchar();

    return 0;
}