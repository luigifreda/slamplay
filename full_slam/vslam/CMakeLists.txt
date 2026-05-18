add_library(vslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp)

# target_include_directories(vslam
#         PUBLIC
#         ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(vslam
        PUBLIC
        ${CORELIBS})
