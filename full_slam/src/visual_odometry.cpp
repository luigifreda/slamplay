//
// Created by gaoxiang on 19-5-4.
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay
//

#include "myslam/visual_odometry.h"
#include "myslam/config.h"

#include <chrono>
#include <opencv2/opencv.hpp>

using namespace slamplay;

namespace myslam {

VisualOdometry::VisualOdometry(const std::string& config_path, const DatasetType& dataset_type)
    : config_file_path_(config_path), dataset_type_(dataset_type) {}

bool VisualOdometry::Init() {
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    const double Ts = 1.0 / Config::Get<double>("Camera.fps");
    timerFps_ = std::make_unique<ChronoFps>(Ts);

    switch (dataset_type_)
    {
        case DatasetType::KITTI:
            dataset_ = Dataset::Ptr(new DatasetKitti(Config::Get<std::string>("dataset_dir"), Config::Get<bool>("use_half_resolution")));
            break;

        case DatasetType::EUROC:
            dataset_ = Dataset::Ptr(new DatasetEuroc(Config::Get<std::string>("dataset_dir"), Config::Get<bool>("use_half_resolution")));
            break;

        default:
            LOG(FATAL) << "didn't specify a supported dataset!";
            break;
    }

    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    viewer_->SetMap(map_);

    return true;
}

void VisualOdometry::Run() {
    while (1) {
        if (Step() == false) {
            usleep(5000);
        }
#if 0     
        // step by step 
        cv::waitKey(0);
#else
        // give me a constant time between two calls
        timerFps_->sleep();
#endif
    }

    backend_->Stop();
    viewer_->Close();

    LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    int current_frame_index = dataset_->GetCurrentFrameIndex();
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr)
    {
        usleep(5000);
        return false;
    }
    std::cout << "===========================" << std::endl;
    LOG(INFO) << "Current frame " << current_frame_index << std::endl;
    LOG(INFO) << "VO status: " << frontend_->GetStatusString() << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";

    if (frontend_->GetStatus() == FrontendStatus::LOST) cv::waitKey(0);

    return success;
}

}  // namespace myslam
