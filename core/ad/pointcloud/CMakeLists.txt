include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_pointcloud 
        point_cloud_utils.cpp point_cloud_utils.h lidar_utils.h point_types.h
        cloud_convert.cpp cloud_convert.h
)
target_link_libraries(ad_pointcloud ${CORELIBS})
