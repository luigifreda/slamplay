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

