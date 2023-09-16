include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")

# Locally modify the flags to remove march native optimization for pcl 
# NOTE: There seems to be crash issues when using pcl with march native. 
# This may be due to the fact that PCL available at Debian sources is pre-built without the hardware-specific optimizations, 
# and with C++14. Using different optimizations and C++ standard may result in troubles. 
# https://github.com/PointCloudLibrary/pcl/issues/1684
# related https://github.com/PointCloudLibrary/pcl/issues/619
# https://github.com/ctu-mrs/mrs_lib/discussions/26 
string(REPLACE "-march=native" "" MY_LOCAL_C_FLAGS ${MY_C_FLAGS})
string(REPLACE "-march=native" "" MY_LOCAL_CXX_FLAGS ${MY_CXX_FLAGS})
set(CMAKE_C_FLAGS "${MY_LOCAL_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${MY_LOCAL_CXX_FLAGS}")

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS})
endforeach()