#pragma once

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

#include "string_utils.h"

// a generic set of parameters for specializations of Feature2D
// (some of them may not applicable)
struct FeatureManagerParams {
    std::string type = "orb";
    int normType = cv::NORM_HAMMING;

    const int nfeatures = 500;
    const float scaleFactor = 1.2;
    const int nlevels = 8;

    const int hessianThreshold = 100;  // for SURF
};

// small factory for Feature2D (use default params)
cv::Ptr<cv::Feature2D> getFeature2D(std::string type = "orb") {
    type = str_tolower(type);
    cv::Ptr<cv::Feature2D> fdetector;
    if (type == "orb")
        fdetector = cv::ORB::create();
    else if (type == "brisk")
        fdetector = cv::BRISK::create();
    else if (type == "akaze")
        fdetector = cv::AKAZE::create();
    else if (type == "sift")
        fdetector = cv::SIFT::create();
#ifdef HAVE_OPENCV_CONTRIB
    else if (type == "surf")
        fdetector = cv::xfeatures2d::SURF::create();
#endif
    else
        throw std::runtime_error("Invalid descriptor");
    return fdetector;
}

// small factory for Feature2D (use default params)
cv::Ptr<cv::Feature2D> getFeature2D(const FeatureManagerParams& params) {
    std::string type = str_tolower(params.type);
    cv::Ptr<cv::Feature2D> fdetector;
    if (type == "orb")
        fdetector = cv::ORB::create(params.nfeatures, params.scaleFactor, params.nlevels);
    else if (type == "brisk")
        fdetector = cv::BRISK::create();  // leave it decide
    else if (type == "akaze")
        fdetector = cv::AKAZE::create();  // leave it decide
    else if (type == "sift")
        fdetector = cv::SIFT::create(params.nfeatures, params.nlevels);
#ifdef HAVE_OPENCV_CONTRIB
    else if (type == "surf")
        fdetector = cv::xfeatures2d::SURF::create(params.hessianThreshold, params.nlevels);
#endif
    else
        throw std::runtime_error("Invalid descriptor");
    return fdetector;
}