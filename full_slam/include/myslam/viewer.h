//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include <thread>
#include <pangolin/pangolin.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

/**
 * Visualization
 */
class Viewer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    void SetMap(Map::Ptr map) { map_ = map; }

    void Close();

    // Add a current frame
    void AddCurrentFrame(Frame::Ptr current_frame);

    // Update map
    void UpdateMap();

   private:
    void ThreadLoop();

    void DrawFrame(Frame::Ptr frame, const float* color);
    void DrawActiveKeyFrames();
    void DrawAllKeyFrames();        
    void DrawFrames(const std::unordered_map<unsigned long, Frame::Ptr>& keyframes,
                    const float color[3]);

    void DrawTrajectory();
    
    void DrawActiveMapPoints();
    void DrawAllMapPoints();    
    void DrawPoints(const std::unordered_map<unsigned long, MapPoint::Ptr>& landmarks,
                    const float color[3], GLfloat point_size=3); 

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    std::unordered_map<unsigned long, Frame::Ptr> all_keyframes_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
    std::unordered_map<unsigned long, MapPoint::Ptr> all_landmarks_;    
    bool map_updated_ = false;

    double camera_scale_ = 1.0; 

    std::mutex viewer_data_mutex_;
};
}  // namespace myslam

