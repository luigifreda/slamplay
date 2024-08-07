/**
 * To test the VINO api, and the base function of HFNet
 *
 * Result:
======================================
Evaluate the run time perfomance in dataset:
test parameter, nFeatures: 300, fThreshold: 0.005000, nNMSRadius: 4
test dataset: TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/

Only detect the local keypoints:
run costs: 14.0173 ± 1.08526
copy costs: 0.073487 ± 0.02281
select costs: 0.712338 ± 0.0953048
NMS costs: 1.38445 ± 0.368556
top costs: 0.00413998 ± 0.00110398
resampler costs: 0.360226 ± 0.0266516
detect costs: 16.5651 ± 1.25615

Detect the full features with intermediate:
run costs: 14.8936 ± 0.319231
copy costs: 0.0750034 ± 0.00427455
select costs: 0.786258 ± 0.101484
NMS costs: 1.51293 ± 0.396017
top costs: 0.00439961 ± 0.00127086
resampler costs: 0.385109 ± 0.0244027
detect costs: 17.8576 ± 0.583938
run global costs: 8.86505 ± 0.141589
detect global costs: 9.02595 ± 0.141241

Detect the full features:
run costs: 22.7289 ± 0.331593
copy costs: 0.0745052 ± 0.00711539
select costs: 0.873383 ± 0.10625
NMS costs: 1.68873 ± 0.431953
top costs: 0.0049334 ± 0.001891
resampler costs: 0.428852 ± 0.0236922
detect costs: 25.8261 ± 0.643804

Detect the local features with HFextractor [kImageToLocal]:
detect costs: 16.3739 ± 0.579497

Detect the local features with HFextractor [kImageToLocalAndIntermediate]:
detect costs: 17.8502 ± 0.54653
detect global costs: 9.03912 ± 0.139315
 */
#include <dirent.h>
#include <boost/format.hpp>
#include <chrono>
#include <fstream>
#include <random>

#include <cv/matches_utils.h>
#include "Frame.h"
#include "HFNetSettings.h"
#include "features_dl/hfnet/HFNetVINOModel.h"

#ifdef USE_OPENVINO

using namespace cv;
using namespace std;
using namespace hfnet;
using namespace ov;

HFNetSettings *settings;
HFNetVINOModel *pModelImageToLocalAndGlobal;
HFNetVINOModel *pModelImageToLocal;
HFNetVINOModel *pModelImageToLocalAndInter;
HFNetVINOModel *pModelInterToGlobal;

TicToc timerDetect;
TicToc timerRun;
TicToc timerCopy;
TicToc timerSelect;
TicToc timerNMS;
TicToc timerTop;
TicToc timerResampler;
TicToc timerDetectGlobal;
TicToc timerRunGlobal;

void ClearTimer() {
    timerDetect.clearBuff();
    timerRun.clearBuff();
    timerCopy.clearBuff();
    timerSelect.clearBuff();
    timerNMS.clearBuff();
    timerTop.clearBuff();
    timerResampler.clearBuff();
    timerDetectGlobal.clearBuff();
    timerRunGlobal.clearBuff();
}

void PrintTimer() {
    if (!timerRun.empty()) cout << "run costs: " << timerRun.aveCost() << " ± " << timerRun.devCost() << endl;
    if (!timerCopy.empty()) cout << "copy costs: " << timerCopy.aveCost() << " ± " << timerCopy.devCost() << endl;
    if (!timerSelect.empty()) cout << "select costs: " << timerSelect.aveCost() << " ± " << timerSelect.devCost() << endl;
    if (!timerNMS.empty()) cout << "NMS costs: " << timerNMS.aveCost() << " ± " << timerNMS.devCost() << endl;
    if (!timerTop.empty()) cout << "top costs: " << timerTop.aveCost() << " ± " << timerTop.devCost() << endl;
    if (!timerResampler.empty()) cout << "resampler costs: " << timerResampler.aveCost() << " ± " << timerResampler.devCost() << endl;
    if (!timerDetect.empty()) cout << "detect costs: " << timerDetect.aveCost() << " ± " << timerDetect.devCost() << endl;
    if (!timerRunGlobal.empty()) cout << "run global costs: " << timerRunGlobal.aveCost() << " ± " << timerRunGlobal.devCost() << endl;
    if (!timerDetectGlobal.empty()) cout << "detect global costs: " << timerDetectGlobal.aveCost() << " ± " << timerDetectGlobal.devCost() << endl;
}

void Mat2Tensor(const cv::Mat &mat, ov::Tensor *tensor) {
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), tensor->data<float>());
    mat.convertTo(fromMat, CV_32F);
}

void Tensor2Mat(ov::Tensor *tensor, cv::Mat &mat) {
    const cv::Mat fromTensor(cv::Size(tensor->get_shape()[1], tensor->get_shape()[2]), CV_32FC(tensor->get_shape()[3]), tensor->data<float>());
    fromTensor.convertTo(mat, CV_32F);
}

void ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output) {
    const int batch_size = data.get_shape()[0];
    const int data_height = data.get_shape()[1];
    const int data_width = data.get_shape()[2];
    const int data_channels = data.get_shape()[3];

    output = cv::Mat(warp.get_shape()[0], data_channels, CV_32F);

    const int num_sampling_points = warp.get_shape()[0];
    if (num_sampling_points > 0)
    {
        Resampler(data.data<float>(), warp.data<float>(), output.ptr<float>(),
                  batch_size, data_height, data_width,
                  data_channels, num_sampling_points);
    }
}

bool DetectImageToLocal(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    timerCopy.Toc();

    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tScoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");

    const int width = tScoreDense.get_shape()[2], height = tScoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tScoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tScoreDense.get_shape()[1] - 1.f);

    timerSelect.Tic();
    auto vResScoresDense = tScoreDense.data<float>();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }
    timerSelect.Toc();

    timerNMS.Tic();
    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);
    timerNMS.Toc();

    if (vKeyPoints.size() > nKeypointsNum)
    {
        timerTop.Tic();
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::nth_element(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
        timerTop.Toc();
    }

    timerResampler.Tic();
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tLocalDescriptorMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectImageToLocalAndGlobal(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                                 int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    timerCopy.Toc();

    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tScoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");
    ov::Tensor tGlobalDescriptor = pModel->mInferRequest->get_tensor("global_descriptor");

    auto vResGlobalDescriptor = tGlobalDescriptor.data<float>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }

    const int width = tScoreDense.get_shape()[2], height = tScoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tScoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tScoreDense.get_shape()[1] - 1.f);

    timerSelect.Tic();
    auto vResScoresDense = tScoreDense.data<float>();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }
    timerSelect.Toc();

    timerNMS.Tic();
    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);
    timerNMS.Toc();

    if (vKeyPoints.size() > nKeypointsNum)
    {
        timerTop.Tic();
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::nth_element(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
        timerTop.Toc();
    }

    timerResampler.Tic();
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tLocalDescriptorMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectImageToLocalAndInter(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &preGlobalDescriptors,
                                int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    timerCopy.Toc();

    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tScoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");
    ov::Tensor tIntermediate = pModel->mInferRequest->get_tensor("pred/MobilenetV2/expanded_conv_6/input:0");

    Tensor2Mat(&tIntermediate, preGlobalDescriptors);

    const int width = tScoreDense.get_shape()[2], height = tScoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tScoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tScoreDense.get_shape()[1] - 1.f);

    timerSelect.Tic();
    auto vResScoresDense = tScoreDense.data<float>();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }
    timerSelect.Toc();

    timerNMS.Tic();
    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);
    timerNMS.Toc();

    if (vKeyPoints.size() > nKeypointsNum)
    {
        timerTop.Tic();
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::nth_element(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
        timerTop.Toc();
    }

    timerResampler.Tic();
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tLocalDescriptorMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectInterToGlobal(HFNetVINOModel *pModel, const cv::Mat &preGlobalDescriptors, cv::Mat &globalDescriptors) {
    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != preGlobalDescriptors.cols || inputShape[1] != preGlobalDescriptors.rows || inputShape[3] != preGlobalDescriptors.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(preGlobalDescriptors, &inputTensor);

    timerRunGlobal.Tic();
    pModel->mInferRequest->infer();
    timerRunGlobal.Toc();

    ov::Tensor tGlobalDescriptor = pModel->mInferRequest->get_tensor("global_descriptor");

    auto vResGlobalDescriptor = tGlobalDescriptor.data<float>();
    globalDescriptors = cv::Mat::zeros(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }

    return true;
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const std::string strLocalModelPath("/media/microsd/ubuntu18/Work/slam_rgbd_wss/HFNet_SLAM/model/hfnet_vino_local_f32/");
const std::string strGlobalModelPath("/media/microsd/ubuntu18/Work/slam_rgbd_wss/HFNet_SLAM/model/hfnet_vino_global_f32/");
const std::string strFullModelPath("/media/microsd/ubuntu18/Work/slam_rgbd_wss/HFNet_SLAM/model/hfnet_vino_full_f32/");

int nFeatures = 300;  // 1500~326, 1000~217
float fThreshold = 0.005;
int nNMSRadius = 4;

int main(int argc, char *argv[]) {
    settings = new HFNetSettings(strSettingsPath, 0);
    pModelImageToLocal = new HFNetVINOModel(strLocalModelPath + "/saved_model.xml", strLocalModelPath + "/saved_model.bin", kImageToLocal, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelImageToLocalAndInter = new HFNetVINOModel(strLocalModelPath + "/saved_model.xml", strLocalModelPath + "/saved_model.bin", kImageToLocalAndIntermediate, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelInterToGlobal = new HFNetVINOModel(strGlobalModelPath + "/saved_model.xml", strGlobalModelPath + "/saved_model.bin", kIntermediateToGlobal, {1, settings->newImSize().height / 8, settings->newImSize().width / 8, 96});
    pModelImageToLocalAndGlobal = new HFNetVINOModel(strFullModelPath + "/saved_model.xml", strFullModelPath + "/saved_model.bin", kImageToLocalAndGlobal, {1, settings->newImSize().height, settings->newImSize().width, 1});

    pModelImageToLocal->PrintInputAndOutputsInfo();
    pModelInterToGlobal->PrintInputAndOutputsInfo();

    vector<string> files = GetPngFiles(strDatasetPath);  // get all image files

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, globalDescriptors, preGlobalDescriptors;

    // randomly detect an image and show the results
    char command = ' ';
    int select = 0;
    while (1)
    {
        if (command == 'x')
            break;
        else if (command == 's')
            select = std::max(select - 1, 0);
        else if (command == 'w')
            select += 1;
        else if (command == 'q')
            nFeatures = std::max(nFeatures - 20, 0);
        else if (command == 'e')
            nFeatures += 20;
        else if (command == 'a')
            fThreshold = std::max(fThreshold - 0.005, 0.005);
        else if (command == 'd')
            fThreshold += 0.005;
        else if (command == 'z')
            nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c')
            nNMSRadius += 1;
        else
            select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "nFeatures: " << nFeatures << endl;
        cout << "fThreshold: " << fThreshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        ClearTimer();
        timerDetect.Tic();
        // DetectImageToLocalAndInter(pModelImageToLocalAndInter, image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius);
        if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius))
            cerr << "error while detecting!" << endl;
        timerDetect.Toc();
        timerDetectGlobal.Tic();
        // DetectInterToGlobal(pModelInterToGlobal, preGlobalDescriptors, globalDescriptors);
        if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
            cerr << "error while detecting!" << endl;
        timerDetectGlobal.Toc();
        cout << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();

        cout << localDescriptors.ptr<float>()[0] << endl;
        cout << preGlobalDescriptors.row(50).col(50) << endl;
        cout << globalDescriptors.col(0).rowRange(100, 110) << endl;

        showKeypoints("press 'x' for further test", image, vKeyPoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    cout << "======================================" << endl
         << "Evaluate the run time perfomance in dataset: " << endl
         << (boost::format("test parameter, nFeatures: %d, fThreshold: %f, nNMSRadius: %d") % nFeatures % fThreshold % nNMSRadius).str() << endl
         << "test dataset: " << strDatasetPath << endl;

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocal(pModelImageToLocal, image, vKeyPoints, localDescriptors, nFeatures, fThreshold, nNMSRadius);
            timerDetect.Toc();
        }
        cout << "Only detect the local keypoints: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocalAndInter(pModelImageToLocalAndInter, image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius);
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            DetectInterToGlobal(pModelInterToGlobal, preGlobalDescriptors, globalDescriptors);
            timerDetectGlobal.Toc();
        }
        cout << "Detect the full features with intermediate: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocalAndGlobal(pModelImageToLocalAndGlobal, image, vKeyPoints, localDescriptors, globalDescriptors, nFeatures, fThreshold, nNMSRadius);
            timerDetect.Toc();
        }
        cout << "Detect the full features: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            if (!pModelImageToLocal->Detect(image, vKeyPoints, localDescriptors, nFeatures, fThreshold, nNMSRadius))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
        }
        cout << "Detect the local features with HFextractor [kImageToLocal]: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string &file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
                cerr << "error while detecting!" << endl;
            timerDetectGlobal.Toc();
        }
        cout << "Detect the local features with HFextractor [kImageToLocalAndIntermediate]: " << endl;
        PrintTimer();
    }

    cout << endl
         << "Press 'ENTER' to exit" << endl;
    getchar();

    return 0;
}

#else  // USE_OPENVINO

int main() {
    cerr << "You must set USE_OPENVINO in CMakeLists.txt to enable OpenVINO function." << endl;
    return -1;
}

#endif  // USE_OPENVINO