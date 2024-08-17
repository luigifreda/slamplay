#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include "features_dl/hfnet/HFNetBaseModel.h"

namespace hfnet {

class HFNetBaseModel;

class HFextractor {
   public:
    HFextractor(int nfeatures, float threshold, HFNetBaseModel *pModels);

    HFextractor(int nfeatures, float threshold, float scaleFactor,
                int nlevels, const std::vector<HFNetBaseModel *> &vpModels);

    ~HFextractor() {}

    // Compute the features and descriptors on an image.
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint> &_keypoints,
                   cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors);

    int inline GetLevels(void) {
        return nlevels;
    }

    float inline GetScaleFactor(void) {
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors(void) {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(void) {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(void) {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(void) {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<int> mnFeaturesPerLevel;

    int nfeatures;
    float threshold;

    std::vector<HFNetBaseModel *> mvpModels;

   protected:
    double scaleFactor;
    int nlevels;
    bool bUseOctTree;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<int> umax;

    void ComputePyramid(const cv::Mat &image);

    int ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);

    int ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);

    int ExtractMultiLayersParallel(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                                   cv::Mat &localDescriptors, cv::Mat &globalDescriptors);
};

}  // namespace hfnet

#endif
