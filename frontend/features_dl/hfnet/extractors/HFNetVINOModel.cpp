#include "extractors/HFNetVINOModel.h"

using namespace cv;
using namespace std;

namespace hfnet {

#ifdef USE_OPENVINO

using namespace ov;

ov::Core HFNetVINOModel::core;

HFNetVINOModel::HFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath, ModelDetectionMode mode, const cv::Vec4i inputShape) {
    mStrXmlPath = strXmlPath;
    mStrBinPath = strBinPath;
    mbVaild = LoadHFNetVINOModel(strXmlPath, strBinPath);

    mMode = mode;
    mInputShape = {(size_t)inputShape(0), (size_t)inputShape(1), (size_t)inputShape(2), (size_t)inputShape(3)};
    if (mMode == kImageToLocalAndGlobal)
    {
        mvOutputTensorNames.emplace_back("pred/local_head/detector/Squeeze:0");
        mvOutputTensorNames.emplace_back("local_descriptor_map");
        mvOutputTensorNames.emplace_back("global_descriptor");
    } else if (mMode == kImageToLocal)
    {
        mvOutputTensorNames.emplace_back("pred/local_head/detector/Squeeze:0");
        mvOutputTensorNames.emplace_back("local_descriptor_map");
    } else if (mMode == kImageToLocalAndIntermediate)
    {
        mvOutputTensorNames.emplace_back("pred/local_head/detector/Squeeze:0");
        mvOutputTensorNames.emplace_back("local_descriptor_map");
        mvOutputTensorNames.emplace_back("pred/MobilenetV2/expanded_conv_6/input:0");
    } else if (mMode == kIntermediateToGlobal)
    {
        mvOutputTensorNames.emplace_back("global_descriptor");
    } else
    {
        mbVaild = false;
        return;
    }

    const ov::Layout modelLayout{"NHWC"};

    mpModel->reshape({{mpModel->input().get_any_name(), mInputShape}});

    ov::preprocess::PrePostProcessor ppp(mpModel);

    ppp.input()
        .tensor()
        .set_layout(modelLayout);
    ppp.input().model().set_layout(modelLayout);
    // ppp.output(0).tensor().set_element_type(ov::element::f32);
    // ppp.output(1).tensor().set_element_type(ov::element::f32);
    mpModel = ppp.build();

    mpExecutableNet = make_shared<ov::CompiledModel>(core.compile_model(mpModel, "CPU"));

    mInferRequest = make_shared<ov::InferRequest>(mpExecutableNet->create_infer_request());

    mInputTensor = mInferRequest->get_input_tensor();
    mbVaild = true;
}

bool HFNetVINOModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                            int nKeypointsNum, float threshold) {
    if (mMode != kImageToLocalAndIntermediate) return false;

    Mat2Tensor(image, &mInputTensor);

    if (!Run()) return false;

    Tensor2Mat(&mvNetResults[2], globalDescriptors);
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], vKeyPoints, localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetVINOModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                            int nKeypointsNum, float threshold) {
    if (mMode != kImageToLocal) return false;

    vKeyPoints.clear();

    Mat2Tensor(image, &mInputTensor);

    if (!Run()) return false;
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], vKeyPoints, localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetVINOModel::Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) {
    if (mMode != kIntermediateToGlobal) return false;

    Mat2Tensor(intermediate, &mInputTensor);
    if (!Run()) return false;
    GetGlobalDescriptorFromTensor(mvNetResults[0], globalDescriptors);
    return true;
}

bool HFNetVINOModel::Run(void) {
    if (!mbVaild) return false;

    if (mpExecutableNet->input().get_shape() != mInputTensor.get_shape()) return false;

    mInferRequest->set_input_tensor(mInputTensor);
    mInferRequest->infer();
    mvNetResults.clear();
    for (const auto &ouputName : mvOutputTensorNames)
    {
        mvNetResults.emplace_back(mInferRequest->get_tensor(ouputName));
    }

    return true;
}

void HFNetVINOModel::GetLocalFeaturesFromTensor(const ov::Tensor &tScoreDense, const ov::Tensor &tDescriptorsMap,
                                                std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                                int nKeypointsNum, float threshold) {
    const int width = tScoreDense.get_shape()[2], height = tScoreDense.get_shape()[1];
    const float scaleWidth = (tDescriptorsMap.get_shape()[2] - 1.f) / (float)(tScoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tDescriptorsMap.get_shape()[1] - 1.f) / (float)(tScoreDense.get_shape()[1] - 1.f);

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

    vKeyPoints = NMS(vKeyPoints, width, height, 4);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::nth_element(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
    }

    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tDescriptorsMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
}

void HFNetVINOModel::GetGlobalDescriptorFromTensor(const ov::Tensor &tDescriptors, cv::Mat &globalDescriptors) {
    auto vResGlobalDescriptor = tDescriptors.data<float>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }
}

bool HFNetVINOModel::LoadHFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath) {
    mpModel = core.read_model(strXmlPath, strBinPath);
    return true;
}

void HFNetVINOModel::PrintInputAndOutputsInfo(void) {
    std::cout << "model name: " << mpModel->get_friendly_name() << std::endl;

    const std::vector<ov::Output<ov::Node>> inputs = mpModel->inputs();
    for (const ov::Output<ov::Node> input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<ov::Node>> outputs = mpModel->outputs();
    for (const ov::Output<ov::Node> output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

void HFNetVINOModel::Mat2Tensor(const cv::Mat &mat, ov::Tensor *tensor) {
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), tensor->data<float>());
    mat.convertTo(fromMat, CV_32F);
}

void HFNetVINOModel::Tensor2Mat(ov::Tensor *tensor, cv::Mat &mat) {
    const cv::Mat fromTensor(cv::Size(tensor->get_shape()[1], tensor->get_shape()[2]), CV_32FC(tensor->get_shape()[3]), tensor->data<float>());
    fromTensor.convertTo(mat, CV_32F);
}

void HFNetVINOModel::ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output) {
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

#endif  // USE_OPENVINO

}  // namespace hfnet
