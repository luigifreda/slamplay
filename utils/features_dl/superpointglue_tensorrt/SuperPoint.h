//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include <tensorrtbuffers/buffers.h>
#include "SuperPointGlueConfig.h"

using tensorrt_common::TensorRTUniquePtr;

class SuperPoint {
   public:
    explicit SuperPoint(SuperPointConfig super_point_config);

    bool build();

    bool infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

    void visualization(const std::string &image_name, const cv::Mat &image);

    void save_engine();

    bool deserialize_engine();

    void getInputOutputNames();

   private:
    SuperPointConfig super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<double>> descriptors_;

    std::map<std::string, std::string> inOut_;  //!< Input and output mapping of the network
    std::map<std::string, nvinfer1::Dims> inOutDims_;
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime_;

    bool isVerbose_{false};

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool process_input(const tensorrt_buffers::BufferManager &buffers, const cv::Mat &image);

    bool process_output(const tensorrt_buffers::BufferManager &buffers, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

    void remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border, int height,
                        int width);

    std::vector<size_t> sort_indexes(std::vector<float> &data);

    void top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k);

    void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints, int h, int w,
                               double threshold);

    void sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                            std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s = 8);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif  // SUPER_POINT_H_
