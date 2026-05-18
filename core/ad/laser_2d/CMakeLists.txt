include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_laser_2d 
        frame.cpp frame.h
        icp_2d.cpp icp_2d.h
        lidar_2d_utils.cpp lidar_2d_utils.h
        likelihood_field.cpp likelihood_field.h
        loop_closing.cpp loop_closing.h
        mapping_2d.cpp mapping_2d.h
        multi_resolution_likelihood_field.cpp multi_resolution_likelihood_field.h
        occupancy_map.cpp occupancy_map.h
        submap.cpp submap.h
        )
target_link_libraries(ad_laser_2d ${CORELIBS} ad_common ad_pointcloud)
