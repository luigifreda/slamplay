//
// Created by gaoxiang on 19-5-4.
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay
//
#pragma once

#include "myslam/backend.h"
#include "myslam/common_include.h"
#include "myslam/dataset.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"

#include "time/ChronoFps.h"

#include <memory>

namespace myslam {

/**
 * VO External Interface
 */
class VisualOdometry {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(const std::string& config_path, const DatasetType& dataset_type = DatasetType::KITTI);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// Get frontend status
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

   private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
    DatasetType dataset_type_;

    std::unique_ptr<slamplay::ChronoFps> timerFps_;
};

}  // namespace myslam
