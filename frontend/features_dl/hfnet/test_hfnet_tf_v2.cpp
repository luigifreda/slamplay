/**
 * To test the tensorflow api, and the base function of HFNet
 *
 * Result:
 *
======================================
Evaluate the run time perfomance in dataset:
test parameter, nFeatures: 300, fThreshold: 0.005000, nNMSRadius: 4
test dataset: /media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/

Only detect the local features:
run costs: 4.7159 ± 0.306045
copy costs: 0.0443026 ± 0.00564807
select costs: 0.50105 ± 0.0286744
top costs: 0.00395379 ± 0.00115836
resampler costs: 0.345047 ± 0.0279108
detect costs: 5.61314 ± 0.326196

Detect the full features:
run costs: 6.87347 ± 0.298149
copy costs: 0.0485791 ± 0.012615
select costs: 0.521006 ± 0.0408057
top costs: 0.0039382 ± 0.00113337
resampler costs: 0.36118 ± 0.0419838
detect costs: 7.8165 ± 0.330191

Detect the full features with intermediate:
run costs: 4.84551 ± 0.108042
copy costs: 0.0556787 ± 0.0035558
select costs: 0.520267 ± 0.0322422
top costs: 0.00395235 ± 0.00112911
resampler costs: 0.352219 ± 0.0307679
detect costs: 5.90465 ± 0.12739
run global costs: 2.68353 ± 0.0545445
detect global costs: 2.764 ± 0.0564719

Detect the full features with pModel [kImageToLocalAndGlobal]:
detect costs: 7.78245 ± 0.138222

Detect the local features with HFextractor [kImageToLocal]:
detect costs: 5.66649 ± 0.126189

Detect the local features with HFextractor [kImageToLocalAndIntermediate]:
detect costs: 5.96151 ± 0.118566
detect global costs: 2.79904 ± 0.0640896

 */
#include <dirent.h>
#include <boost/format.hpp>
#include <chrono>
#include <fstream>
#include <random>

#include <cv/matches_utils.h>
#include "Frame.h"
#include "HFNetSettings.h"
#include "features_dl/hfnet/HFNetTFModelV2.h"

#ifdef USE_TENSORFLOW

using namespace cv;
using namespace std;
using namespace hfnet;
using namespace tensorflow;

HFNetSettings *settings;
HFNetTFModelV2 *pModelImageToLocalAndGlobal;
HFNetTFModelV2 *pModelImageToLocal;
HFNetTFModelV2 *pModelImageToLocalAndInter;
HFNetTFModelV2 *pModelInterToGlobal;

TicToc timerDetect;
TicToc timerRun;
TicToc timerCopy;
TicToc timerSelect;
TicToc timerTop;
TicToc timerResampler;
TicToc timerDetectGlobal;
TicToc timerRunGlobal;

void ClearTimer() {
    timerDetect.clearBuff();
    timerRun.clearBuff();
    timerCopy.clearBuff();
    timerSelect.clearBuff();
    timerTop.clearBuff();
    timerResampler.clearBuff();
    timerDetectGlobal.clearBuff();
    timerRunGlobal.clearBuff();
}

void PrintTimer() {
    if (!timerRun.empty()) cout << "run costs: " << timerRun.aveCost() << " ± " << timerRun.devCost() << endl;
    if (!timerCopy.empty()) cout << "copy costs: " << timerCopy.aveCost() << " ± " << timerCopy.devCost() << endl;
    if (!timerSelect.empty()) cout << "select costs: " << timerSelect.aveCost() << " ± " << timerSelect.devCost() << endl;
    if (!timerTop.empty()) cout << "top costs: " << timerTop.aveCost() << " ± " << timerTop.devCost() << endl;
    if (!timerResampler.empty()) cout << "resampler costs: " << timerResampler.aveCost() << " ± " << timerResampler.devCost() << endl;
    if (!timerDetect.empty()) cout << "detect costs: " << timerDetect.aveCost() << " ± " << timerDetect.devCost() << endl;
    if (!timerRunGlobal.empty()) cout << "run global costs: " << timerRunGlobal.aveCost() << " ± " << timerRunGlobal.devCost() << endl;
    if (!timerDetectGlobal.empty()) cout << "detect global costs: " << timerDetectGlobal.aveCost() << " ± " << timerDetectGlobal.devCost() << endl;
}

void Mat2Tensor(const cv::Mat &mat, tensorflow::Tensor *tensor) {
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), tensor->flat<float>().data());
    mat.convertTo(fromMat, CV_32F);
}

void Tensor2Mat(tensorflow::Tensor *tensor, cv::Mat &mat) {
    const cv::Mat fromTensor(cv::Size(tensor->shape().dim_size(1), tensor->shape().dim_size(2)), CV_32FC(tensor->shape().dim_size(3)), tensor->flat<float>().data());
    mat = fromTensor.clone();
}

void ResamplerTF(const tensorflow::Tensor &data, const tensorflow::Tensor &warp, cv::Mat &output) {
    const tensorflow::TensorShape &data_shape = data.shape();
    const int batch_size = data_shape.dim_size(0);
    const int data_height = data_shape.dim_size(1);
    const int data_width = data_shape.dim_size(2);
    const int data_channels = data_shape.dim_size(3);
    const tensorflow::TensorShape &warp_shape = warp.shape();

    tensorflow::TensorShape output_shape = warp.shape();
    // output_shape.set_dim(output_shape.dims() - 1, data_channels);
    // output = Tensor(DT_FLOAT, output_shape);
    output = cv::Mat(output_shape.dim_size(0), data_channels, CV_32F);

    const int num_sampling_points = warp.NumElements() / batch_size / 2;
    if (num_sampling_points > 0)
    {
        Resampler(data.flat<float>().data(),
                  warp.flat<float>().data(), output.ptr<float>(), batch_size,
                  data_height, data_width, data_channels, num_sampling_points);
    }
}

bool DetectImageToLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    timerCopy.Toc();

    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocal->mSession->Run({
                                                          {"image", tImage},
                                                          {"pred/simple_nms/radius", tRadius},
                                                      },
                                                      {"scores_dense_nms", "local_descriptor_map"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>();  // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    timerSelect.Tic();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense(row * width + col);
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

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectImageToLocalAndGlobal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                                 int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    static Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    timerCopy.Toc();

    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocalAndGlobal->mSession->Run({
                                                                   {"image", tImage},
                                                                   {"pred/simple_nms/radius", tRadius},
                                                               },
                                                               {"scores_dense_nms", "local_descriptor_map", "global_descriptor"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>();  // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();
    auto vResGlobalDescriptor = outputs[2].tensor<float, 2>();

    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor(temp);
    }

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    timerSelect.Tic();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense(row * width + col);
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

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectImageToLocalAndInter(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &preGlobalDescriptors,
                                int nKeypointsNum, float threshold, int nRadius) {
    timerCopy.Tic();
    TicToc timer;
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    timerCopy.Toc();

    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocalAndInter->mSession->Run({
                                                                  {"image", tImage},
                                                                  {"pred/simple_nms/radius", tRadius},
                                                              },
                                                              {"scores_dense_nms", "local_descriptor_map", "pred/MobilenetV2/expanded_conv_6/input:0"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>();  // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();

    Tensor2Mat(&outputs[2], preGlobalDescriptors);

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    timerSelect.Tic();
    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense(row * width + col);
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

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    timerResampler.Toc();

    return true;
}

bool DetectInterToGlobal(const cv::Mat &preGlobalDescriptors, cv::Mat &globalDescriptors) {
    Tensor tPreGlobalDescriptors(DT_FLOAT, TensorShape({1, preGlobalDescriptors.rows, preGlobalDescriptors.cols, preGlobalDescriptors.channels()}));
    Mat2Tensor(preGlobalDescriptors, &tPreGlobalDescriptors);

    timerRunGlobal.Tic();
    vector<Tensor> outputs;
    Status status = pModelInterToGlobal->mSession->Run({
                                                           {"pred/MobilenetV2/expanded_conv_6/input:0", tPreGlobalDescriptors},
                                                       },
                                                       {"global_descriptor"}, {}, &outputs);
    timerRunGlobal.Toc();
    if (!status.ok()) return false;

    auto vResGlobalDescriptor = outputs[0].tensor<float, 2>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F, outputs[0].flat<float>().data()).clone();

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

const std::string strTFModelPath("/media/microsd/ubuntu18/Work/slam_rgbd_wss/HFNet_SLAM/model/hfnet_tf_v2_NMS2/");

int nFeatures = 300;  // 1500~326, 1000~217
float fThreshold = 0.005;
int nNMSRadius = 4;

int main(int argc, char *argv[]) {
    settings = new HFNetSettings(strSettingsPath, 0);
    pModelImageToLocalAndGlobal = new HFNetTFModelV2(strTFModelPath, kImageToLocalAndGlobal, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelImageToLocal = new HFNetTFModelV2(strTFModelPath, kImageToLocal, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelImageToLocalAndInter = new HFNetTFModelV2(strTFModelPath, kImageToLocalAndIntermediate, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelInterToGlobal = new HFNetTFModelV2(strTFModelPath, kIntermediateToGlobal, {1, settings->newImSize().height / 8, settings->newImSize().width / 8, 96});

    vector<string> files = GetPngFiles(strDatasetPath);  // get all image files

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, preGlobalDescriptors, globalDescriptors;

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
        DetectImageToLocalAndInter(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius);
        // if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius))
        //     cerr << "error while detecting!" << endl;
        timerDetect.Toc();
        timerDetectGlobal.Tic();
        DetectInterToGlobal(preGlobalDescriptors, globalDescriptors);
        // if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
        //     cerr << "error while detecting!" << endl;
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
            DetectImageToLocal(image, vKeyPoints, localDescriptors, nFeatures, fThreshold, nNMSRadius);
            timerDetect.Toc();
        }
        cout << "Only detect the local features: " << endl;
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
            DetectImageToLocalAndGlobal(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius);
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
            DetectImageToLocalAndInter(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold, nNMSRadius);
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            DetectInterToGlobal(preGlobalDescriptors, globalDescriptors);
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
            if (!pModelImageToLocalAndGlobal->Detect(image, vKeyPoints, localDescriptors, globalDescriptors, nFeatures, fThreshold))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
        }
        cout << "Detect the full features with pModel [kImageToLocalAndGlobal]: " << endl;
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
            if (!pModelImageToLocal->Detect(image, vKeyPoints, localDescriptors, nFeatures, fThreshold))
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
            if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, nFeatures, fThreshold))
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

#else  // USE_TENSORFLOW

int main() {
    cerr << "You must set USE_TENSORFLOW in CMakeLists.txt to enable TensorFlow function." << endl;
    return -1;
}

#endif  // USE_TENSORFLOW