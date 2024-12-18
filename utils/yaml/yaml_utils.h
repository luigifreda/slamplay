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

#include "io/messages.h"
#include "io/utils_stdio.h"

#include <math.h>
#include <cstring>
#include <filesystem>
#include <limits>

#include <yaml-cpp/yaml.h>  //libyaml-cpp-dev

namespace slamplay {

template <typename T>
inline T getParam(const YAML::Node& node, const std::string& name, const T& defaultValue, std::string parent = "") {
    if (!node.IsDefined()) {
        MSG_ERROR_STREAM("Cannot find YAML node ");
        return defaultValue;
    }
    std::string fullname = parent.empty() ? name : parent + std::string(":") + name;
    std::string::size_type colons_pos = name.find(':');
    if (colons_pos == std::string::npos) {
        if (node[name]) {
            std::cout << "Found parameter: " << fullname << ", value: " << node[name] << std::endl;
            return node[name].as<T>();
        } else {
            std::cout << "Cannot find value for parameter: " << fullname << ", assigning default: " << defaultValue
                      << std::endl;
        }
    } else {
        std::string group = name.substr(0, colons_pos);
        std::string param = name.substr(colons_pos + 1);
        return getParam<T>(node[group], param, defaultValue, group);
    }
    return defaultValue;
}

template <typename T>
inline std::vector<T> getArray(const YAML::Node& config, const std::string& name, const std::vector<T> defaultValue) {
    std::vector<T> outArray;

    // Check if the node exists and is a sequence
    if (config[name] && config[name].IsSequence()) {
        // Iterate through the sequence and add each element to the vector
        for (std::size_t i = 0; i < config[name].size(); ++i) {
            outArray.push_back(config[name][i].as<T>());
        }
        // Output the contents of the vector
        std::cout << "Found array of parameters:" << name << ", value: " << outArray << std::endl;
        return outArray;
    } else {
        std::cout << "Cannot find value for parameter: " << name << ", assigning default: " << defaultValue
                  << std::endl;
        return defaultValue;
    }
}

}  // namespace slamplay