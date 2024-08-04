#pragma once

#include "macros.h"
#include "yaml_utils.h"

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

class DepthAnythingSettings {
   public:
    static const std::string kDataDir;
    static const std::string kDepthAnythingConfigFile;

   public:
    DepthAnythingSettings(const std::string& configFile = kDepthAnythingConfigFile);

    friend std::ostream& operator<<(std::ostream& output, const DepthAnythingSettings& s);

    float depthScale() { return depthScale_; }
    std::string strModelPath() { return strModelPath_; }
    std::string strDatasetPath() { return datasetPath_; }
    const std::vector<float>& cameraParams() { return cameraParams_; }

    cv::Size imageSize() { return imageSize_; }

   private:
    bool readConfig(const std::string& configFile);

   private:
    std::string strModelPath_;
    cv::Size imageSize_;
    float depthScale_ = -1;

    std::string datasetPath_;
    std::vector<float> cameraParams_;
};
