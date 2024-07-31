#include "extractors/HFNetTFModel.h"
namespace hfnet {

#ifdef USE_TENSORFLOW

using namespace cv;
using namespace std;
using namespace tensorflow;

bool HFNetTFModel::mbLoadedResampler = false;

HFNetTFModel::HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir) {
    bool bLoadedLib = LoadResamplerOp(strResamplerDir);
    bool bLoadedModel = LoadHFNetTFModel(strModelDir);

    mbVaild = bLoadedLib & bLoadedModel;
}

void HFNetTFModel::Compile(const cv::Vec4i inputSize, bool onlyDetectLocalFeatures) {
    // The tensorflow model cost huge time at the first detection.
    // Therefore, give a fake image to compile
    // The size of fake image should be the same as the real image.

    Mat fakeImg(inputSize(2), inputSize(1), CV_8UC1);
    cv::randu(fakeImg, Scalar(0), Scalar(255));
    vector<tensorflow::Tensor> vNetResults;
    Run(fakeImg, vNetResults, onlyDetectLocalFeatures, 1000, 0.01);
}

bool HFNetTFModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                          int nKeypointsNum, float threshold) {
    Run(image, mvNetResults, false, nKeypointsNum, threshold);
    GetGlobalDescriptorFromTensor(mvNetResults[3], globalDescriptors);
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], mvNetResults[2], vKeyPoints, localDescriptors);
    return true;
}

bool HFNetTFModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                          int nKeypointsNum, float threshold) {
    Run(image, mvNetResults, true, nKeypointsNum, threshold);
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], mvNetResults[2], vKeyPoints, localDescriptors);
    return true;
}

bool HFNetTFModel::Run(const cv::Mat &image, std::vector<tensorflow::Tensor> &vNetResults, bool onlyDetectLocalFeatures,
                       int nKeypointsNum, float threshold) {
    if (!mbVaild) return false;

    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tThreshold(DT_FLOAT, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tThreshold.scalar<float>()() = threshold;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);

    std::vector<string> outputTensorName = {"keypoints:0", "local_descriptors:0", "scores:0"};
    if (!onlyDetectLocalFeatures) outputTensorName.emplace_back("global_descriptor:0");
    Status status = mSession->Run({{"image:0", tImage},
                                   {"pred/top_k_keypoints/k", tKeypointsNum},
                                   {"pred/keypoint_extraction/GreaterEqual/y", tThreshold}},
                                  outputTensorName, {}, &vNetResults);
    return status.ok();
}

void HFNetTFModel::GetLocalFeaturesFromTensor(const tensorflow::Tensor &tKeyPoints, const tensorflow::Tensor &tDescriptorsMap, const tensorflow::Tensor &tScoreDense,
                                              std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors) {
    int nResNumber = tKeyPoints.shape().dim_size(1);

    auto vResKeypoints = tKeyPoints.tensor<int32, 3>();
    auto vResLocalDes = tDescriptorsMap.tensor<float, 2>();
    auto vResScores = tScoreDense.tensor<float, 3>();

    vKeyPoints.clear();
    vKeyPoints.reserve(nResNumber);
    localDescriptors = cv::Mat(nResNumber, 256, CV_32F);
    KeyPoint kp;
    kp.angle = 0;
    kp.octave = 0;
    for (int index = 0; index < nResNumber; index++)
    {
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeyPoints.emplace_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp);
        }
    }
}

void HFNetTFModel::GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors) {
    auto vResGlobalDescriptor = tDescriptors.tensor<float, 2>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor(temp);
    }
}

bool HFNetTFModel::LoadResamplerOp(const std::string &strResamplerDir) {
    if (mbLoadedResampler) return true;
    TF_Status *status = TF_NewStatus();
    TF_LoadLibrary(strResamplerDir.c_str(), status);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "TF_LoadLibrary() error with code: " << TF_GetCode(status) << std::endl;
        std::cerr << "Failed to load resampler.so in path: " << strResamplerDir << std::endl;
        return false;
    }
    std::cout << "Sucessfully loaded resampler.so" << std::endl;
    mbLoadedResampler = true;
    return true;
}

bool HFNetTFModel::LoadHFNetTFModel(const std::string &strModelDir) {
    mstrTFModelPath = strModelDir;
    tensorflow::Status status;
    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions runOptions;
    tensorflow::SavedModelBundle bundle;

    status = LoadSavedModel(sessionOptions, runOptions, strModelDir, {tensorflow::kSavedModelTagServe}, &bundle);
    if (!status.ok()) {
        std::cerr << "Failed to load HFNet model at path: " << strModelDir << std::endl;
        return false;
    }

    mSession = std::move(bundle.session);
    status = mSession->Create(mGraph);
    if (!status.ok()) {
        std::cerr << "Failed to create mGraph for HFNet" << std::endl;
        return false;
    }

    return true;
}

void HFNetTFModel::Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor) {
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
}

#endif

}  // namespace hfnet