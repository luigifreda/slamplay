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
#include <math.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "macros.h"
#include "yaml/yaml_utils.h"

using namespace slamplay;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
std::string configFile = dataDir + "/config.yaml";

struct YamlConfig {
    std::string inputFilePath;
    struct SubParameters {
        double param1 = 0.0;
        std::string param2;
        bool param3 = false;
    } subParameters;
    std::vector<int> arrayInt;
    std::vector<float> arrayFloat;
};

bool readYamlConfig(const YAML::Node& node, YamlConfig& config) {
    MSG_INFO_STREAM("Reading yaml configuration");
    config.inputFilePath = getParam(node, std::string("input_file_path"), std::string());
    config.subParameters.param1 = getParam(node, "subparameters:param1", config.subParameters.param1);
    config.subParameters.param2 = getParam(node, "subparameters:param2", config.subParameters.param2);
    config.subParameters.param3 = getParam(node, "subparameters:param3", config.subParameters.param3);
    config.arrayInt = getArray<int>(node, "arrayInt", config.arrayInt);
    config.arrayFloat = getArray<float>(node, "arrayFloat", config.arrayFloat);
    return true;
}

int main(int argc, char** argv) {
    std::cout << "reading " << configFile << std::endl;
    auto node = YAML::LoadFile(configFile);
    if (!node) {
        MSG_ERROR("Didn't find config file. Exiting.");
        return 0;
    }

    const auto nodeTest = node["Test"];
    if (!nodeTest) {
        MSG_ERROR("Didn't find Test node in config file. Exiting.");
        return 0;
    }

    MSG_INFO_STREAM("Reading yaml configuration");
    YamlConfig config;
    if (!readYamlConfig(nodeTest, config)) {
        MSG_ERROR("Could not read yaml config. Exiting.");
        return 0;
    }
    return 0;
}
