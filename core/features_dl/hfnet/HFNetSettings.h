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

#include "features_dl/hfnet/HFextractor.h"

#include "macros.h"
#include "yaml/yaml_utils.h"

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
