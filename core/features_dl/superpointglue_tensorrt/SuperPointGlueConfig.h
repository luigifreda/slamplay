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
#ifndef READ_CONFIG_H_
#define READ_CONFIG_H_

#include <yaml-cpp/yaml.h>
#include <iostream>

#include "io/file_utils.h"

struct SuperPointConfig {
    int max_keypoints{};
    double keypoint_threshold{};
    int remove_borders{};
    int dla_core{};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct SuperGlueConfig {
    int image_width{};
    int image_height{};
    int dla_core{};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct Configs {
    std::string model_dir;

    SuperPointConfig superpoint_config;
    SuperGlueConfig superglue_config;

    Configs(const std::string& config_file, const std::string& model_dir) {
        std::cout << "Config file is " << config_file << std::endl;
        if (!slamplay::fileExists(config_file)) {
            std::cerr << "Config file " << config_file << " doesn't exist." << std::endl;
            return;
        }
        YAML::Node file_node = YAML::LoadFile(config_file);

        YAML::Node superpoint_node = file_node["superpoint"];
        superpoint_config.max_keypoints = superpoint_node["max_keypoints"].as<int>();
        superpoint_config.keypoint_threshold = superpoint_node["keypoint_threshold"].as<double>();
        superpoint_config.remove_borders = superpoint_node["remove_borders"].as<int>();
        superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();
        YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
        size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
        for (size_t i = 0; i < superpoint_num_input_tensor_names; i++) {
            superpoint_config.input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
        }
        YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
        size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
        for (size_t i = 0; i < superpoint_num_output_tensor_names; i++) {
            superpoint_config.output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
        }
        std::string superpoint_onnx_file = superpoint_node["onnx_file"].as<std::string>();
        std::string superpoint_engine_file = superpoint_node["engine_file"].as<std::string>();
        superpoint_config.onnx_file = slamplay::concatenateFolderAndFileName(model_dir, superpoint_onnx_file);
        superpoint_config.engine_file = slamplay::concatenateFolderAndFileName(model_dir, superpoint_engine_file);

        YAML::Node superglue_node = file_node["superglue"];
        superglue_config.image_width = superglue_node["image_width"].as<int>();
        superglue_config.image_height = superglue_node["image_height"].as<int>();
        superglue_config.dla_core = superglue_node["dla_core"].as<int>();
        YAML::Node superglue_input_tensor_names_node = superglue_node["input_tensor_names"];
        size_t superglue_num_input_tensor_names = superglue_input_tensor_names_node.size();
        for (size_t i = 0; i < superglue_num_input_tensor_names; i++) {
            superglue_config.input_tensor_names.push_back(superglue_input_tensor_names_node[i].as<std::string>());
        }
        YAML::Node superglue_output_tensor_names_node = superglue_node["output_tensor_names"];
        size_t superglue_num_output_tensor_names = superglue_output_tensor_names_node.size();
        for (size_t i = 0; i < superglue_num_output_tensor_names; i++) {
            superglue_config.output_tensor_names.push_back(superglue_output_tensor_names_node[i].as<std::string>());
        }
        std::string superglue_onnx_file = superglue_node["onnx_file"].as<std::string>();
        std::string superglue_engine_file = superglue_node["engine_file"].as<std::string>();
        superglue_config.onnx_file = slamplay::concatenateFolderAndFileName(model_dir, superglue_onnx_file);
        superglue_config.engine_file = slamplay::concatenateFolderAndFileName(model_dir, superglue_engine_file);
    }
};

#endif  // READ_CONFIG_H_
