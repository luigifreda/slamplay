
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "features_dl/hfnet/HFextractor.h"

using namespace cv;
using namespace std;

namespace hfnet {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

HFextractor::HFextractor(int _nfeatures, float _threshold, BaseModel *_pModels) : nfeatures(_nfeatures), threshold(_threshold) {
    mvpModels.resize(1);
    mvpModels[0] = _pModels;
    scaleFactor = 1.0;
    nlevels = 1;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    // This is for orientation
    //  pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

HFextractor::HFextractor(int _nfeatures, float _threshold, float _scaleFactor,
                         int _nlevels, const std::vector<BaseModel *> &_vpModels) : nfeatures(_nfeatures), threshold(_threshold), mvpModels(_vpModels) {
    scaleFactor = _scaleFactor;
    nlevels = _nlevels;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    // This is for orientation
    //  pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

int HFextractor::operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                            cv::Mat &localDescriptors, cv::Mat &globalDescriptors) {
    if (image.empty() || image.type() != CV_8UC1) return -1;

    int res = -1;
    if (nlevels == 1)
        res = ExtractSingleLayer(image, vKeyPoints, localDescriptors, globalDescriptors);
    else
    {
        if (mvpModels[0]->Type() == kHFNetVINOModel)
            res = ExtractMultiLayers(image, vKeyPoints, localDescriptors, globalDescriptors);
        else
            res = ExtractMultiLayersParallel(image, vKeyPoints, localDescriptors, globalDescriptors);
    }
    return res;
}

void HFextractor::ComputePyramid(const cv::Mat &image) {
    mvImagePyramid[0] = image;
    for (int level = 1; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));

        // Compute the resized image
        if (level != 0)
        {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
        }
    }
}

int HFextractor::ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                                    cv::Mat &localDescriptors, cv::Mat &globalDescriptors) {
    if (!mvpModels[0]->Detect(image, vKeyPoints, localDescriptors, globalDescriptors, nfeatures, threshold))
        cerr << "Error while detecting keypoints" << endl;

    return vKeyPoints.size();
}

int HFextractor::ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                                    cv::Mat &localDescriptors, cv::Mat &globalDescriptors) {
    ComputePyramid(image);

    int nKeypoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        if (level == 0)
        {
            if (!mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], globalDescriptors, mnFeaturesPerLevel[level], threshold))
                cerr << "Error while detecting keypoints" << endl;
        } else
        {
            if (!mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], mnFeaturesPerLevel[level], threshold))
                cerr << "Error while detecting keypoints" << endl;
        }
        nKeypoints += allKeypoints[level].size();
    }
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);

    return vKeyPoints.size();
}
class DetectParallel : public cv::ParallelLoopBody {
   public:
    DetectParallel(vector<cv::KeyPoint> *allKeypoints, cv::Mat *allDescriptors, cv::Mat *globalDescriptors, HFextractor *pExtractor)
        : mAllKeypoints(allKeypoints), mAllDescriptors(allDescriptors), mGlobalDescriptors(globalDescriptors), mpExtractor(pExtractor) {}

    virtual void operator()(const cv::Range &range) const CV_OVERRIDE {
        for (int level = range.start; level != range.end; ++level)
        {
            if (level == 0)
            {
                if (!mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], *mGlobalDescriptors, mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold))
                    cerr << "Error while detecting keypoints" << endl;
            } else
            {
                if (!mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold))
                    cerr << "Error while detecting keypoints" << endl;
            }
        }
    }

    DetectParallel &operator=(const DetectParallel &) {
        return *this;
    };

   private:
    vector<cv::KeyPoint> *mAllKeypoints;
    cv::Mat *mAllDescriptors;
    cv::Mat *mGlobalDescriptors;
    HFextractor *mpExtractor;
};

int HFextractor::ExtractMultiLayersParallel(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                                            cv::Mat &localDescriptors, cv::Mat &globalDescriptors) {
    ComputePyramid(image);

    int nKeypoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);

    DetectParallel detector(allKeypoints.data(), allDescriptors.data(), &globalDescriptors, this);
    cv::parallel_for_(cv::Range(0, nlevels), detector);

    for (int level = 0; level < nlevels; ++level)
        nKeypoints += allKeypoints[level].size();

    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);

    return vKeyPoints.size();
}

}  // namespace hfnet