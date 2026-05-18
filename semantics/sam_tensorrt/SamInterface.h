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
#pragma once

#include <macros.h>
#include "io/file_utils.h"
#include "io/messages.h"
#include "sam_tensorrt/Sam.h"
#include "sam_tensorrt/sam_export.h"
#include "sam_tensorrt/sam_utils.h"

#include "tensorrt/tensorrt_utils.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

class SamInterface {
   public:
    static constexpr int kAutoSegmentStep = 40;
    static constexpr double kAutoSegmentIouThreshold = 0.86;
    static constexpr double kMinArea = 100;

    const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
    const std::string kSamDataDir = kDataDir + "/segment_anything";
    const std::string kSamModelsDir = kSamDataDir + "/models";
    const std::string kSamEmbeddingModelFile = kSamModelsDir + "/vit_l_embedding.engine";
    const std::string kSamModelFile = kSamModelsDir + "/sam_onnx_example.engine";

   public:
    SamInterface() = default;
    ~SamInterface() = default;

    int loadModels();
    at::Tensor& processEmbedding(const cv::Mat& frame);
    cv::Mat processAutoSegment(const cv::Mat& frame);
    cv::Mat processSinglePointMask(const cv::Mat& frame, const int x, const int y);
    cv::Mat showAutoSegmentResult(const cv::Mat& frame, const cv::Mat& mask, cv::Mat& outImage);

   protected:
    bool exportEngine_(const std::string& engineFile, bool isEmbedding);
    nvinfer1::ICudaEngine* deserializeEngine_(const std::string& engineFile);
    nvinfer1::ICudaEngine* loadEngineWithRecovery_(const std::string& engineFile, bool isEmbedding);
    void exportEnginesIfNeeded_();
    cv::Mat autoSegment_(const cv::Mat& image);

   public:
    std::shared_ptr<SamEmbedding> engImgEmbedding;
    std::shared_ptr<SamPromptEncoderAndMaskDecoder> engPromptEncAndMaskDec;
    at::Tensor image_embedding;

    nvinfer1::IRuntime* runtime{nullptr};
};

inline bool SamInterface::exportEngine_(const std::string& engineFile, bool isEmbedding) {
    const std::string onnxFile = slamplay::getFileNameWithouExtension(engineFile) + ".onnx";
    if (!slamplay::fileExists(onnxFile)) {
        MSG_ERROR("File not found: " << onnxFile);
        return false;
    }

    MSG_WARN_STREAM("Exporting " << engineFile << " from onnx model, it may take some time...");
    if (isEmbedding) {
        export_engine_sam_image_encoder(onnxFile, engineFile);
    } else {
        export_engine_sam_sample_encoder_and_mask_decoder(onnxFile, engineFile);
    }

    if (!slamplay::fileExists(engineFile)) {
        MSG_WARN_STREAM("engine export did not create: " << engineFile);
        return false;
    }

    return true;
}

inline nvinfer1::ICudaEngine* SamInterface::deserializeEngine_(const std::string& engineFile) {
    std::ifstream engineStream(engineFile, std::ifstream::binary);
    if (!engineStream.good()) {
        MSG_WARN_STREAM("failed to read model: " << engineFile);
        return nullptr;
    }

    engineStream.seekg(0, engineStream.end);
    const auto fileSize = static_cast<size_t>(engineStream.tellg());
    engineStream.seekg(0, engineStream.beg);
    std::vector<char> engineData(fileSize);
    engineStream.read(engineData.data(), fileSize);

    if (engineStream) {
        std::cout << "all characters read successfully: " << engineFile << std::endl;
    } else {
        std::cout << "error: only " << engineStream.gcount() << " could be read" << std::endl;
        return nullptr;
    }
    engineStream.close();

#if NV_TENSORRT_VERSION_CODE < 100000L
    return runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
#else
    return runtime->deserializeCudaEngine(engineData.data(), fileSize);
#endif
}

inline nvinfer1::ICudaEngine* SamInterface::loadEngineWithRecovery_(const std::string& engineFile, bool isEmbedding) {
    nvinfer1::ICudaEngine* engine = deserializeEngine_(engineFile);
    if (engine) {
        return engine;
    }

    MSG_WARN_STREAM("failed to deserialize engine: " << engineFile);
    std::error_code ec;
    std::filesystem::remove(engineFile, ec);
    if (ec) {
        MSG_WARN_STREAM("failed to remove incompatible engine cache " << engineFile << ": " << ec.message());
    }

    if (!exportEngine_(engineFile, isEmbedding)) {
        return nullptr;
    }

    engine = deserializeEngine_(engineFile);
    if (!engine) {
        MSG_WARN_STREAM("failed to deserialize rebuilt engine: " << engineFile);
    }
    return engine;
}

void SamInterface::exportEnginesIfNeeded_() {
    std::ifstream f1(kSamEmbeddingModelFile);
    if (!f1.good())
    {
        exportEngine_(kSamEmbeddingModelFile, true);
    }

    std::ifstream f2(kSamModelFile);
    if (!f2.good())
    {
        exportEngine_(kSamModelFile, false);
    }
}

int SamInterface::loadModels() {
    exportEnginesIfNeeded_();

    runtime = nvinfer1::createInferRuntime(slamplay::NvLogger::instance);
    if (!runtime) {
        MSG_ERROR("failed to create runtime");
        return -1;
    }

    {
        // Load SAM embedding model
        nvinfer1::ICudaEngine* engine = loadEngineWithRecovery_(kSamEmbeddingModelFile, true);
        if (!engine) {
            MSG_WARN_STREAM("failed to load engine: " << kSamEmbeddingModelFile);
            return -1;
        }
        engImgEmbedding = std::shared_ptr<SamEmbedding>(new SamEmbedding(std::to_string(1), engine));
    }

    {
        // Load SAM model
        nvinfer1::ICudaEngine* engine = loadEngineWithRecovery_(kSamModelFile, false);
        if (!engine) {
            MSG_WARN_STREAM("failed to load engine: " << kSamModelFile);
            return -1;
        }

        engPromptEncAndMaskDec = std::shared_ptr<SamPromptEncoderAndMaskDecoder>(new SamPromptEncoderAndMaskDecoder(std::to_string(1), engine));
    }

    MSG_INFO("Successfully loaded models");

    return 1;
}

at::Tensor& SamInterface::processEmbedding(const cv::Mat& frame) {
    auto res = engImgEmbedding->prepareInput(frame);
    std::cout << "------------------prepareInput: " << res << std::endl;
    res = engImgEmbedding->infer();
    std::cout << "------------------infer: " << res << std::endl;
    image_embedding = engImgEmbedding->verifyOutput();
    std::cout << "------------------verifyOutput: " << std::endl;

    return image_embedding;
}

cv::Mat SamInterface::processSinglePointMask(const cv::Mat& frame, const int x, const int y) {
    auto res = engPromptEncAndMaskDec->prepareInput(x, y, frame, image_embedding);
    // std::vector<int> mult_pts = {x,y,x-5,y-5,x+5,y+5};
    // auto res = engPromptEncAndMaskDec->prepareInput(mult_pts, image_embeddings);
    std::cout << "------------------prepareInput: " << res << std::endl;
    res = engPromptEncAndMaskDec->infer();
    std::cout << "------------------infer: " << res << std::endl;
    cv::Mat mask;
    engPromptEncAndMaskDec->verifyOutput(mask);
    mask *= 255;
    std::cout << "------------------verifyOutput: " << std::endl;
    return mask;
}

cv::Mat SamInterface::processAutoSegment(const cv::Mat& frame) {
    auto mask = autoSegment_(frame);
    return mask;
}

cv::Mat SamInterface::showAutoSegmentResult(const cv::Mat& frame, const cv::Mat& mask, cv::Mat& outImage) {
    const double overlayFactor = 0.5;
    const int maxMaskValue = 255 * (1 - overlayFactor);
    if (outImage.empty()) outImage = cv::Mat::zeros(frame.size(), CV_8UC3);

    static std::map<int, cv::Vec3b> colors;

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            auto value = (int)mask.at<double>(i, j);
            if (value <= 0) {
                continue;
            }

            auto it = colors.find(value);
            if (it == colors.end()) {
                colors.insert(it, {value, cv::Vec3b(rand() % maxMaskValue, rand() % maxMaskValue,
                                                    rand() % maxMaskValue)});
            }

            outImage.at<cv::Vec3b>(i, j) = it->second + frame.at<cv::Vec3b>(i, j) * overlayFactor;
        }
    }

#if 1
    // draw circles on the image to indicate the sample points
    const int step = SamInterface::kAutoSegmentStep;
    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            cv::circle(outImage, {j * step, i * step}, 1, {0, 0, 255}, -1);
        }
    }
#endif

    return outImage;
}

// Just a poor version of
// https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
cv::Mat SamInterface::autoSegment_(const cv::Mat& image) {
    cv::Size numPoints = {image.cols / kAutoSegmentStep, image.rows / kAutoSegmentStep};

    const auto size = image.size();
    cv::Mat mask, outImage = cv::Mat::zeros(size, CV_64FC1);

    std::vector<double> masksAreas;
    float iou;

    const auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numPoints.height; i++) {
        for (int j = 0; j < numPoints.width; j++) {
            cv::Point input(cv::Point((j + 0.5) * size.width / numPoints.width,
                                      (i + 0.5) * size.height / numPoints.height));

            // double iou;
            // m_model->getMask({input}, {}, {}, mask, iou);
            // if (mask.empty() || iou < kAutoSegmentIouThreshold) {
            // continue;
            // }

            auto res = engPromptEncAndMaskDec->prepareInput(input.x, input.y, image, image_embedding);
            res = engPromptEncAndMaskDec->infer();
            cv::Mat mask;
            engPromptEncAndMaskDec->verifyOutput(mask, &iou);
            if (mask.empty() || iou < kAutoSegmentIouThreshold) {
                continue;
            }
            mask *= 255;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                continue;
            }

            int maxContourIndex = 0;
            double maxContourArea = 0;
            for (int i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxContourArea) {
                    maxContourArea = area;
                    maxContourIndex = i;
                }
            }
            if (maxContourArea < kMinArea) {
                continue;
            }

            cv::Mat contourMask = cv::Mat::zeros(size, CV_8UC1);
            cv::drawContours(contourMask, contours, maxContourIndex, cv::Scalar(255), cv::FILLED);
            cv::Rect boundingBox = cv::boundingRect(contours[maxContourIndex]);

            int index = masksAreas.size() + 1, numPixels = 0;
            for (int i = boundingBox.y; i < boundingBox.y + boundingBox.height; i++) {
                for (int j = boundingBox.x; j < boundingBox.x + boundingBox.width; j++) {
                    if (contourMask.at<uchar>(i, j) == 0) {
                        continue;
                    }

                    auto dst = (int)outImage.at<double>(i, j);
                    if (dst > 0 && masksAreas[dst - 1] < maxContourArea) {
                        continue;
                    }
                    outImage.at<double>(i, j) = index;
                    numPixels++;
                }
            }
            if (numPixels == 0) {
                continue;
            }

            masksAreas.emplace_back(maxContourArea);
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
    return outImage;
}