#ifndef BASEMODEL_H
#define BASEMODEL_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace hfnet {

enum ModelType {
    kHFNetTFModel,
    kHFNetRTModel,
    kHFNetVINOModel,
};
const std::string gStrModelTypeName[] = {"kHFNetTFModel", "kHFNetRTModel", "kHFNetVINOModel"};

enum ModelDetectionMode {
    kImageToLocalAndGlobal,
    kImageToLocal,
    kImageToLocalAndIntermediate,
    kIntermediateToGlobal
};
const std::string gStrModelDetectionName[] = {"ImageToLocalAndGlobal", "ImageToLocal", "ImageToLocalAndIntermediate", "IntermediateToGlobal"};

class ExtractorNode {
   public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class BaseModel {
   public:
    virtual ~BaseModel(void) = default;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) = 0;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) = 0;

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) = 0;

    virtual bool IsValid(void) = 0;

    virtual ModelType Type(void) = 0;
};

class HFNetSettings;

void InitAllModels(HFNetSettings *settings);

void InitAllModels(const std::string &strModelPath, ModelType modelType, cv::Size ImSize, int nLevels, float scaleFactor);

std::vector<BaseModel *> GetModelVec(void);

BaseModel *GetGlobalModel(void);

BaseModel *InitTFModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

BaseModel *InitRTModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

BaseModel *InitVINOModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int minX,
                                            const int maxX, const int minY, const int maxY, const int N);

std::vector<cv::KeyPoint> NMS(const std::vector<cv::KeyPoint> &vToDistributeKeys, int width, int height, int radius);

void Resampler(const float *data, const float *warp, float *output,
               const int batch_size, const int data_height,
               const int data_width, const int data_channels, const int num_sampling_points);

}  // namespace hfnet

#endif