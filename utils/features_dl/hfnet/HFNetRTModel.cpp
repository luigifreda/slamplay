#include "features_dl/hfnet/HFNetRTModel.h"

#ifdef USE_TENSORRT
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace nvinfer1;

namespace hfnet {

void RTLogger::log(Severity severity, AsciiChar const *msg) noexcept {
    if (severity > level) return;

    using namespace std;
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING";
            break;
        case Severity::kINFO:
            std::cerr << "INFO";
            break;
        case Severity::kVERBOSE:
            std::cerr << "VERBOSE";
            break;
    }
    std::cerr << ": " << msg << endl;
}

HFNetRTModel::HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape) {
    mStrTRModelDir = strModelDir + "/";
    mMode = mode;
    mInputShape = {inputShape(0), inputShape(1), inputShape(2), inputShape(3)};
    mStrONNXFile = mStrTRModelDir + "HF-Net.onnx";
    mStrCacheFile = mStrTRModelDir + "HF-Net.cache";

    std::string engineFileName = DecideEigenFileName(mStrTRModelDir, mMode, mInputShape);
    if (!LoadEngineFromFile(engineFileName)) {
        std::cout << "HFNetRTModel() - Loading the model and building the engine ... " << std::endl;
        mvValid = LoadHFNetTRModel();
    } else {
        mvValid = true;
    }

    if (!mvValid) return;

    mpBuffers.reset(new BufferManager(mEngine));

    if (mMode == kImageToLocalAndGlobal)
    {
        mvInputTensors.emplace_back(mpBuffers->getHostBuffer("image:0"), mEngine->getTensorShape("image:0"));

        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("scores_dense_nms:0"), mEngine->getTensorShape("scores_dense_nms:0"));
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("local_descriptor_map:0"), mEngine->getTensorShape("local_descriptor_map:0"));
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("global_descriptor:0"), mEngine->getTensorShape("global_descriptor:0"));
    } else if (mMode == kImageToLocal)
    {
        mvInputTensors.emplace_back(mpBuffers->getHostBuffer("image:0"), mEngine->getTensorShape("image:0"));

        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("scores_dense_nms:0"), mEngine->getTensorShape("scores_dense_nms:0"));
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("local_descriptor_map:0"), mEngine->getTensorShape("local_descriptor_map:0"));
    } else if (mMode == kImageToLocalAndIntermediate || mMode == kIntermediateToGlobal)
    {
        mvValid = false;  // not supported
        return;
    } else
    {
        mvValid = false;
        return;
    }

    mvValid = true;
}

bool HFNetRTModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                          int nKeypointsNum, float threshold) {
    if (mMode != kImageToLocalAndGlobal && mMode != kImageToLocalAndIntermediate) return false;

    Mat2Tensor(image, mvInputTensors[0]);

    if (!Run()) return false;

    if (mMode == kImageToLocalAndGlobal)
        GetGlobalDescriptorFromTensor(mvOutputTensors[2], globalDescriptors);
    else
        Tensor2Mat(mvOutputTensors[2], globalDescriptors);
    GetLocalFeaturesFromTensor(mvOutputTensors[0], mvOutputTensors[1], vKeyPoints, localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetRTModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                          int nKeypointsNum, float threshold) {
    if (mMode != kImageToLocal) return false;

    Mat2Tensor(image, mvInputTensors[0]);

    if (!Run()) return false;
    GetLocalFeaturesFromTensor(mvOutputTensors[0], mvOutputTensors[1], vKeyPoints, localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetRTModel::Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) {
    if (mMode != kIntermediateToGlobal) return false;

    Mat2Tensor(intermediate, mvInputTensors[0]);
    if (!Run()) return false;
    GetGlobalDescriptorFromTensor(mvOutputTensors[0], globalDescriptors);
    return true;
}

bool HFNetRTModel::Run(void) {
    if (!mvValid) return false;
    if (mvInputTensors.empty()) return false;

    // Memcpy from host input buffers to device input buffers
    mpBuffers->copyInputToDevice();

    bool status = mContext->executeV2(mpBuffers->getDeviceBindings().data());
    if (!status) return false;

    // Memcpy from device output buffers to host output buffers
    mpBuffers->copyOutputToHost();

    return true;
}

void HFNetRTModel::GetLocalFeaturesFromTensor(const RTTensor &tScoreDense, const RTTensor &tDescriptorsMap,
                                              std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                                              int nKeypointsNum, float threshold) {
    auto vResScoresDense = static_cast<float *>(tScoreDense.data);  // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = static_cast<float *>(tDescriptorsMap.data);

    const int width = tScoreDense.shape.d[2], height = tScoreDense.shape.d[1];
    const float scaleWidth = (tDescriptorsMap.shape.d[2] - 1.f) / (float)(tScoreDense.shape.d[2] - 1.f);
    const float scaleHeight = (tDescriptorsMap.shape.d[1] - 1.f) / (float)(tScoreDense.shape.d[1] - 1.f);

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

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::nth_element(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint &p1, const cv::KeyPoint &p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
    }

    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    cv::Mat tWarp(vKeyPoints.size(), 2, CV_32FC1);
    auto pWarp = tWarp.ptr<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerRT(tDescriptorsMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
}

void HFNetRTModel::GetGlobalDescriptorFromTensor(const RTTensor &tDescriptors, cv::Mat &globalDescriptors) {
    auto vResGlobalDescriptor = static_cast<float *>(tDescriptors.data);
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }
}

bool HFNetRTModel::LoadHFNetTRModel(void) {
    auto builder = unique_ptr<IBuilder>(createInferBuilder(mLogger));
    if (!builder) return false;

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if (!parser) return false;

    auto parsed = parser->parseFromFile(mStrONNXFile.c_str(), 2);
    if (!parsed) return false;
    network->getInput(0)->setDimensions(mInputShape);

    if (mMode == kImageToLocal)
    {
        network->unmarkOutput(*network->getOutput(2));
    }

    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;
    config->setFlag(BuilderFlag::kFP16);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2 << 20);
    std::unique_ptr<ITimingCache> timingCache{nullptr};
    LoadTimingCacheFile(mStrCacheFile, config, timingCache);
    auto serializedEngine = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    if (mLogger.level >= ILogger::Severity::kINFO) PrintInputAndOutputsInfo(network);

    // Save Engine
    std::string engineFileName = DecideEigenFileName(mStrTRModelDir, mMode, mInputShape);
    SaveEngineToFile(engineFileName, serializedEngine);
    std::cout << "Saved HFNetRT engine to: " << engineFileName << std::endl;

    unique_ptr<IRuntime> runtime{createInferRuntime(mLogger)};
    if (!runtime) return false;

    mEngine = shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!mEngine) return false;

    UpdateTimingCacheFile(mStrCacheFile, config, timingCache);

    mContext = shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) return false;

    return true;
}

void HFNetRTModel::LoadTimingCacheFile(const std::string &strFileName, std::unique_ptr<IBuilderConfig> &config, std::unique_ptr<ITimingCache> &timingCache) {
    std::ifstream iFile(strFileName, std::ios::in | std::ios::binary);
    std::vector<char> content;
    if (!iFile)
    {
        cout << "Could not read timing cache from: " << strFileName
             << ". A new timing cache will be generated and written." << std::endl;
        content = std::vector<char>();
    } else
    {
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        content.resize(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        std::cerr << "Loaded " << fsize << " bytes of timing cache from " << strFileName << std::endl;
    }

    timingCache.reset(config->createTimingCache(static_cast<const void *>(content.data()), content.size()));
    config->setTimingCache(*timingCache, false);
}

void HFNetRTModel::UpdateTimingCacheFile(const std::string &strFileName, std::unique_ptr<IBuilderConfig> &config, std::unique_ptr<ITimingCache> &timingCache) {
    std::unique_ptr<nvinfer1::ITimingCache> fileTimingCache{config->createTimingCache(static_cast<const void *>(nullptr), 0)};

    std::ifstream iFile(strFileName, std::ios::in | std::ios::binary);
    if (iFile)
    {
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        std::vector<char> content(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        std::cerr << "Loaded " << fsize << " bytes of timing cache from " << strFileName << std::endl;
        fileTimingCache.reset(config->createTimingCache(static_cast<const void *>(content.data()), content.size()));
        if (!fileTimingCache)
        {
            throw std::runtime_error("Failed to create timingCache from " + strFileName + "!");
        }
    }
    fileTimingCache->combine(*timingCache, false);
    std::unique_ptr<nvinfer1::IHostMemory> blob{fileTimingCache->serialize()};
    if (!blob)
    {
        throw std::runtime_error("Failed to serialize ITimingCache!");
    }
    std::ofstream oFile(strFileName, std::ios::out | std::ios::binary);
    if (!oFile)
    {
        std::cerr << "Could not write timing cache to: " << strFileName << std::endl;
        return;
    }
    oFile.write((char *)blob->data(), blob->size());
    oFile.close();
    std::cerr << "Saved " << blob->size() << " bytes of timing cache to " << strFileName << std::endl;
}

std::string HFNetRTModel::DecideEigenFileName(const std::string &strEngineSaveDir, ModelDetectionMode mode, const Dims4 inputShape) {
    string strFileName;
    strFileName = gStrModelDetectionName[mode] + "_" +
                  to_string(inputShape.d[0]) + "x" +
                  to_string(inputShape.d[1]) + "x" +
                  to_string(inputShape.d[2]) + "x" +
                  to_string(inputShape.d[3]) + ".engine";
    return strEngineSaveDir + "/" + strFileName;
}

bool HFNetRTModel::SaveEngineToFile(const std::string &strEngineSaveFile, const unique_ptr<IHostMemory> &serializedEngine) {
    std::ofstream engineFile(strEngineSaveFile, std::ios::binary);
    engineFile.write(reinterpret_cast<char const *>(serializedEngine->data()), serializedEngine->size());
    if (engineFile.fail())
    {
        std::cerr << "Saving engine to file failed." << endl;
        return false;
    }
    return true;
}

bool HFNetRTModel::LoadEngineFromFile(const std::string &strEngineSaveFile) {
    std::ifstream engineFile(strEngineSaveFile, std::ios::binary);
    if (!engineFile.good())
    {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << endl;
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> vecEngineBlob(fsize);
    engineFile.read(reinterpret_cast<char *>(vecEngineBlob.data()), fsize);
    if (!engineFile.good())
    {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << endl;
        return false;
    }

    unique_ptr<IRuntime> runtime{createInferRuntime(mLogger)};
    if (!runtime) return false;

    mEngine.reset(runtime->deserializeCudaEngine(vecEngineBlob.data(), vecEngineBlob.size()));
    if (!mEngine) return false;

    mContext = shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) return false;

    return true;
}

void HFNetRTModel::PrintInputAndOutputsInfo(unique_ptr<INetworkDefinition> &network) {
    std::cout << "model name: " << network->getName() << std::endl;

    const std::string strTypeName[] = {"FLOAT", "HALF", "INT8", "INT32", "BOOL", "UINT8"};
    for (int index = 0; index < network->getNbInputs(); ++index)
    {
        std::cout << "    inputs" << std::endl;

        auto input = network->getInput(index);

        std::cout << "        input name: " << input->getName() << std::endl;
        std::cout << "        input type: " << strTypeName[(int)input->getType()] << std::endl;
        std::cout << "        input shape: [";
        for (int alpha = 0; alpha < input->getDimensions().nbDims; ++alpha)
        {
            std::cout << input->getDimensions().d[alpha];
            if (alpha != input->getDimensions().nbDims - 1)
                std::cout << ", ";
            else
                std::cout << "]" << std::endl;
        }
        std::cout << "        input exec: " << input->isExecutionTensor() << endl;
    }

    for (int index = 0; index < network->getNbOutputs(); ++index)
    {
        std::cout << "    outputs" << std::endl;

        auto output = network->getOutput(index);

        std::cout << "        output name: " << output->getName() << std::endl;
        std::cout << "        output type: " << strTypeName[(int)output->getType()] << std::endl;
        std::cout << "        output shape: [";
        for (int alpha = 0; alpha < output->getDimensions().nbDims; ++alpha)
        {
            std::cout << output->getDimensions().d[alpha];
            if (alpha != output->getDimensions().nbDims - 1)
                std::cout << ", ";
            else
                std::cout << "]" << std::endl;
        }
        std::cout << "        output exec: " << output->isExecutionTensor() << endl;
    }
}

void HFNetRTModel::Mat2Tensor(const cv::Mat &mat, RTTensor &tensor) {
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), static_cast<float *>(tensor.data));
    mat.convertTo(fromMat, CV_32F);
}

void HFNetRTModel::Tensor2Mat(const RTTensor &tensor, cv::Mat &mat) {
    const cv::Mat fromTensor(cv::Size(tensor.shape.d[1], tensor.shape.d[2]), CV_32FC(tensor.shape.d[3]), static_cast<float *>(tensor.data));
    fromTensor.convertTo(mat, CV_32F);
}

void HFNetRTModel::ResamplerRT(const RTTensor &data, const cv::Mat &warp, cv::Mat &output) {
    const Dims data_shape = data.shape;
    const int batch_size = data_shape.d[0];
    const int data_height = data_shape.d[1];
    const int data_width = data_shape.d[2];
    const int data_channels = data_shape.d[3];
    const cv::Size warp_shape = warp.size();

    // output_shape.set_dim(output_shape.dims() - 1, data_channels);
    // output = Tensor(DT_FLOAT, output_shape);
    output = cv::Mat(warp.rows, data_channels, CV_32F);

    const int num_sampling_points = warp.size().area() / batch_size / 2;
    if (num_sampling_points > 0)
    {
        Resampler(static_cast<float *>(data.data), warp.ptr<float>(), output.ptr<float>(),
                  batch_size, data_height, data_width,
                  data_channels, num_sampling_points);
    }
}

}  // namespace hfnet

#endif  // USE_TENSORRT
