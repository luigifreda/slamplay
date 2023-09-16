//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//
#pragma once

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;

/**
* Backend
* There is a separate optimization thread, and the optimization is started when the Map is updated
* Map update is triggered by the front end
*/ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    ///Start the optimization thread in the constructor and suspend
    Backend();

    //Set the left and right destination cameras to obtain internal and external parameters
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    ///Set the map
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    ///Trigger map update and start optimization
    void UpdateMap();

    ///Close the backend thread
    void Stop();

   private:
    ///Backend thread
    void BackendLoop();

    ///Optimize the given keyframe and waypoint
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}//namespace myslam
