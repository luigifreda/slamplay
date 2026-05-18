include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(pointcloud_convert 
    packets_parser.cpp packets_parser.h
    velodyne_convertor.cpp velodyne_convertor.h
    velodyne_config.h
)
target_link_libraries(pointcloud_convert ${CORELIBS})
