//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//


#include "myslam/config.h"

namespace myslam {
    
bool Config::SetParameterFile(const std::string &filename) {
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->file_.isOpened() == false) {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        config_->file_.release();
        return false;
    }
    return true;
}

Config::~Config() {
    if (file_.isOpened())
        file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

}
