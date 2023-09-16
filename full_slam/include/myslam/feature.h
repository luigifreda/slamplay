//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame;
struct MapPoint;

/**
* 2D feature points
* will be associated with a map point after triangulation
*/
struct Feature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;//The frame holding the feature
    cv::KeyPoint position_;//2D extraction position
    std::weak_ptr<MapPoint> map_point_;//associated map point

    bool is_outlier_ = false;//Whether it is an abnormal point
    bool is_on_left_image_ = true;//Whether the logo is mentioned on the left picture, false is the right picture

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
};

}//namespace myslam
