add_library(lio_slam SHARED
        frontend.cpp
        keyframe.cpp
        optimization.cpp
        loop_closure.cpp
        localization_fusion.cpp
        map_utils.cpp
)

# target_include_directories(vslam
#         PUBLIC
#         ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(lio_slam
        PUBLIC
        ad_common
        ad_pointcloud
        ad_g2o
        ad_imu
        ad_timer
        ad_laser_3d
        utm_utils
        utm_convert
        ${CORELIBS})
