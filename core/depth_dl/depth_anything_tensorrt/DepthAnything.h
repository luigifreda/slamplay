#pragma once
#include <NvInfer.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Depth_estimation structure
 */
struct DepthEstimation {
    int x;
    int y;
    int label;

    DepthEstimation() {
        x = 0;
        y = 0;
        label = -1;
    }
};

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat &img, int w, int h);

class DepthAnything {
   public:
    DepthAnything(std::string model_path, nvinfer1::ILogger &logger);
    cv::Mat predict(const cv::Mat &image);
    ~DepthAnything();

   private:
    int input_w = 518;
    int input_h = 518;
    float mean[3] = {123.675, 116.28, 103.53};  // [0.485, 0.456, 0.406] * 255
    float std[3] = {58.395, 57.12, 57.375};     // [0.229, 0.224, 0.225] * 255

    std::vector<int> offset;

    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    nvinfer1::INetworkDefinition *network;

    void *buffer[2];
    float *depth_data;
    cudaStream_t stream;

    std::vector<float> preprocess(cv::Mat &image);
    std::vector<DepthEstimation> postprocess(std::vector<int> mask, int img_w, int img_h);
    void build(std::string onnxPath, nvinfer1::ILogger &logger);
    bool saveEngine(const std::string &filename);
};
