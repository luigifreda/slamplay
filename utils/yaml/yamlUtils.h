#pragma once

#include "messages.h"
#include "utilsStdIo.h"

#include <math.h>
#include <cstring>
#include <filesystem>
#include <limits>

#include <yaml-cpp/yaml.h>  //libyaml-cpp-dev

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
