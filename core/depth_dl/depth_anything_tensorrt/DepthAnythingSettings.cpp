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
#include "DepthAnythingSettings.h"

#include <iostream>

using namespace std;
using namespace slamplay;

const std::string DepthAnythingSettings::kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
const std::string DepthAnythingSettings::kDepthAnythingConfigFile = kDataDir + "/depth_anything/config_hypersim.yaml";

DepthAnythingSettings::DepthAnythingSettings(const std::string& configFile) {
    readConfig(configFile);
}

bool DepthAnythingSettings::readConfig(const std::string& configFile) {
    auto node = YAML::LoadFile(configFile.c_str());
    if (!node) {
        MSG_ERROR("Didn't find config file. Exiting.");
        return 0;
    }

    const auto depthAnythingNode = node["DepthAnything"];

    strModelPath_ = kDataDir + "/" + getParam(depthAnythingNode, "modelPath", strModelPath_);
    depthScale_ = getParam(depthAnythingNode, "depthScale", depthScale_);

    const auto datasetNode = node["Dataset"];

    imageSize_.width = getParam(datasetNode, "imageSize:width", imageSize_.width);
    imageSize_.height = getParam(datasetNode, "imageSize:height", imageSize_.height);

    std::string datasetPath = getParam(datasetNode, "path", std::string());
    bool isRelativePath = getParam(datasetNode, "isRelativePath", false);
    datasetPath_ = isRelativePath ? kDataDir + "/" + datasetPath : datasetPath;
    MSG_INFO("Dataset path: " << datasetPath_);

    // read a vector of parameters from the yaml config file
    cameraParams_ = getArray<float>(datasetNode, "cameraParams", cameraParams_);

    if (datasetPath_.empty()) {
        MSG_ERROR("The dataset base path or the dataset sequence name is empty. Exiting.");
        exit(-1);
    }
    return true;
}

ostream& operator<<(std::ostream& output, const DepthAnythingSettings& settings) {
    output << "DepthAnything settings: " << endl;
    output << "\t-Load model path: " << settings.strModelPath_ << endl;
    output << "\t-Depth scale: " << settings.depthScale_ << endl;
    output << "\t-Image size: " << settings.imageSize_ << endl;
    output << "\t-Dataset path: " << settings.datasetPath_ << endl;

    return output;
}
