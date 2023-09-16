//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

//forward declare
struct MapPoint;
struct Feature;

/**
*frame
*Each frame is assigned an independent id, and the key frame is assigned a key frame ID
*/
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    static long factory_id;

    unsigned long id_ = 0;//id of this frame
    unsigned long keyframe_id_ = 0;//id of key frame
    bool is_keyframe_ = false;//Whether it is a key frame
    double time_stamp_;//Timestamp, not used yet
    SE3 pose_;//Tcw form Pose
    std::mutex pose_mutex_;//Pose data lock
    cv::Mat left_img_, right_img_;//stereo images

    //extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    //corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

   public://data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    //set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    ///Set the keyframe and assign and keyframe id
    void SetKeyFrame();

    ///Factory construction mode, assign id
    static std::shared_ptr<Frame> CreateFrame();
};

}//namespace myslam

