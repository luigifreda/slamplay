include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")
#file(GLOB SOURCES "./*.cpp")  # in this folder we do not recursively search in the lib folders

# locally modify the flags to remove march native optimization for gtsam 
# for some reasons test_pointcloud_mapping.cpp crashes when using march=native build option 
# similar problems?: 
#  https://github.com/PointCloudLibrary/pcl/issues/1684
#  https://github.com/PointCloudLibrary/pcl/issues/2364 
# string(REPLACE "-march=native" "" MY_LOCAL_C_FLAGS ${MY_C_FLAGS})
# string(REPLACE "-march=native" "" MY_LOCAL_CXX_FLAGS ${MY_CXX_FLAGS})
# set(CMAKE_C_FLAGS "${MY_LOCAL_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${MY_LOCAL_CXX_FLAGS}")
# NOTE: Problem solved by adding 
# add_definitions(-DPCL_NO_PRECOMPILE) 

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS})
endforeach()