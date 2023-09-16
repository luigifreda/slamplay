//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include <opencv2/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

class Backend;
class Viewer;

enum class FrontendStatus: int { INIT=0, TRACKING_GOOD, TRACKING_BAD, LOST, NUM_STATUS};

/**
* Front end
* Estimate the current frame Pose, add keyframes to the map and trigger optimization when the keyframe conditions are met
*/
class Frontend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    static std::vector<std::string> frontendStatusStrings;  

   public:
    Frontend();
 
    bool AddFrame(Frame::Ptr frame);

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

    FrontendStatus GetStatus() const { return status_; }
    std::string GetStatusString() const {return frontendStatusStrings[static_cast<int>(status_)]; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

   private:
 
    bool Track();

    bool Reset();

    int TrackLastFrame();

    int EstimateCurrentPose();
 
    bool InsertKeyframe();
 
    bool StereoInit();
 
    int DetectFeatures();
 
    int FindFeaturesInRight();
 
    bool BuildInitMap();
 
    int TriangulateNewPoints();
 
    void SetObservationsForKeyFrame();

 
    FrontendStatus status_ = FrontendStatus::INIT;

    Frame::Ptr current_frame_ = nullptr; 
    Frame::Ptr last_frame_ = nullptr; 
    Camera::Ptr camera_left_ = nullptr; 
    Camera::Ptr camera_right_ = nullptr; 

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    SE3 relative_motion_; 

    int tracking_inliers_ = 0; 

 
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 20;
    int num_features_needed_for_keyframe_ = 80;

    int tracking_lk_patch_size = 11; 
    int tracking_lk_max_level = 3; 
 
    cv::Ptr<cv::GFTTDetector> gftt_;   
};

} 

