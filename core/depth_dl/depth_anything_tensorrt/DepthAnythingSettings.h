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

#include "macros.h"
#include "yaml/yaml_utils.h"

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
