#include <dirent.h>
#include <chrono>
#include <fstream>
#include <random>

#include "extractors/HFNetSettings.h"
#include "extractors/HFextractor.h"
#include "features/ORBextractor.h"

#include <cv/matches_utils.h>
#include <io/image_io.h>

#include "macros.h"
#include "messages.h"

using namespace cv;
using namespace std;
using namespace slamplay;
using namespace hfnet;

int main(int argc, char* argv[]) {
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

    vector<string> files = GetPngFiles(strDatasetPath);  // get all image files
    if (files.empty()) {
        cout << "Error, failed to find any valid image in: " << strDatasetPath << endl;
        return 1;
    }
    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << endl;
        return 1;
    }

    const ModelType modelType = kHFNetRTModel;
    // const ModelType modelType = kHFNetTFModel;  // only when tensorflow is available and USE_TENSORFLOW is defined
    InitAllModels(strModelPath, modelType, ImSize, nLevels, scaleFactor);
    auto vpModels = GetModelVec();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, files.size() - 1);

    cv::Mat image;
    vector<KeyPoint> keypoints;
    cv::Mat localDescripotrs, globlaDescriptors;
    vector<int> vLapping = {0, 1000};

    cv::namedWindow("ORB-SLAM");
    cv::moveWindow("ORB-SLAM", 0, 0);
    cv::namedWindow("HFNet-SLAM");
    cv::moveWindow("HFNet-SLAM", 0, 540);

    char command = 0;
    float threshold = 0.01;
    int nFeatures = 1000;
    int select = 0;
    while (select < files.size())
    {
        bool show_key_interation = false;
        if (command == 'x') {
            break;
        } else if (command == 'a') {
            threshold = std::max(threshold - 0.001, 0.005);
            show_key_interation = true;
        } else if (command == 'd') {
            threshold += 0.001;
            show_key_interation = true;
        } else if (command == 's') {
            select = std::max(select - 1, 0);
            show_key_interation = true;
        } else if (command == 'w') {
            select += 1;
            show_key_interation = true;
        } else if (command == 'q') {
            nFeatures = std::max(nFeatures - 200, 0);
            show_key_interation = true;
        } else if (command == 'e') {
            nFeatures += 200;
            show_key_interation = true;
        } else if (command == 'r') {
            select = distribution(generator);
            show_key_interation = true;
        } else {
            select++;
        }
        if (show_key_interation) {
            cout << "command: " << command << endl;
            cout << "select: " << select << endl;
            cout << "nFeatures: " << nFeatures << endl;
            cout << "threshold: " << threshold << endl;
        }

        image = cv::imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        cout << "====================================================" << endl;
        {
            cout << "ORB-SLAM " << endl;
            auto t1 = chrono::steady_clock::now();
            ORBextractor extractor(nFeatures, scaleFactor, 8, 20, 7);
            extractor(image, cv::Mat(), keypoints, localDescripotrs);
            auto t2 = chrono::steady_clock::now();
            auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            showKeypoints("ORB-SLAM", image, keypoints);
            cout << "\t cost time: " << t << " ms" << endl;
            cout << "\t key point number: " << keypoints.size() << endl;
        }

        {
            cout << "HFNet-SLAM " << endl;
            HFextractor extractorHF(nFeatures, threshold, scaleFactor, nLevels, vpModels);
            auto t1 = chrono::steady_clock::now();
            extractorHF(image, keypoints, localDescripotrs, globlaDescriptors);
            auto t2 = chrono::steady_clock::now();
            auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            showKeypoints("HFNet-SLAM", image, keypoints);
            cout << "\t cost time: " << t << " ms" << endl;
            cout << "\t key point number: " << keypoints.size() << endl;
        }

        command = cv::waitKey();
    };

    return 0;
}