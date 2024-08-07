#pragma once

#include "features_dl/hfnet/HFextractor.h"

#include "macros.h"
#include "yaml_utils.h"

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

namespace hfnet {

class HFNetSettings {
   public:
    static const std::string kDataDir;
    static const std::string kHFNetConfigFile;

   public:
    HFNetSettings(const std::string& configFile = kHFNetConfigFile);

    friend std::ostream& operator<<(std::ostream& output, const HFNetSettings& s);

    ModelType modelType() { return modelType_; }
    int nFeatures() { return nFeatures_; }
    int nLevels() { return nLevels_; }
    float scaleFactor() { return scaleFactor_; }
    float threshold() { return threshold_; }
    std::string strModelPath() { return strModelPath_; }
    std::string strDatasetPath() { return datasetPath_; }

    cv::Size imageSize() { return imageSize_; }

   private:
    bool readConfig(const std::string& configFile);

   private:
    ModelType modelType_ = kHFNetRTModel;
    int nFeatures_ = 1000;
    float scaleFactor_ = 1.0;
    int nLevels_ = 1;
    float threshold_ = 0.01;
    std::string strModelPath_;
    cv::Size imageSize_;

    std::string datasetPath_;
};

}  // namespace hfnet
