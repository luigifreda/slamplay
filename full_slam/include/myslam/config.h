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
//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include "myslam/common_include.h"

namespace myslam {

/**
* Configuration class, use SetParameterFile to determine the configuration file
* Then use Get to get the corresponding value
* Singleton mode
*/
class Config {
   private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config() {}//private constructor makes a singleton
   public:
    ~Config();//close the file when deconstructing

    //set a new config file
    static bool SetParameterFile(const std::string &filename);

    //access the parameter values
    template <typename T>
    static T Get(const std::string &key) {
        return static_cast<T>(Config::config_->file_[key]);
    }

    static bool IsAvailable(const std::string &key) {
        return !Config::config_->file_[key].empty();
    }

    static cv::FileStorage& File() {
        return Config::config_->file_;
    }

};

template <>
inline bool Config::Get<bool>(const std::string &key) {
    std::string val = static_cast<std::string>(Config::config_->file_[key]);
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    return (val=="true");        
}

}//namespace myslam

