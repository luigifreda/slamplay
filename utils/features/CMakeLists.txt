
include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(orbslam2_features ORBextractor.cpp)
target_link_libraries(orbslam2_features ${CORELIBS})
