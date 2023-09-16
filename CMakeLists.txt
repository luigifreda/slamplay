cmake_minimum_required(VERSION 3.15)
project(slamplay)

#====================================
# options 

set(OPENCV_VERSION          "4" CACHE STRING "Desired OpenCV version") # this is the first target version that is searched for 
set(SET_RPATH               ON  CACHE BOOL   "Set RPATH instead of the default RUNPATH") # for solving the linking issue with g2o binaries (see the related node in the compiler options section)
set(BUILD_WITH_MARCH_NATIVE ON  CACHE BOOL   "Build with \"-march native\"")
set(WITH_OPENMP             ON  CACHE BOOL    "Add OpenMP support") # useful for PCL (since we use the macro PCL_NO_PRECOMPILE!)

# NOTE: this is to remove the tons of cmake warning coming from importing ext packages 
# see here: https://github.com/PointCloudLibrary/pcl/issues/3680
# when this is fixed, we can remove the following 3 lines.
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

#====================================
# c++ standard 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # CMake will generate an error if it is unable to compile the code with a compiler that supports C++17.

#====================================
# compiler options 

# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_BUILD_TYPE Debug) # Uncomment this to force Debug build

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release 
    set(MY_FLAGS "-Wall -Wextra -O3 -fPIC -DNDEBUG -Wno-unused-parameter -Wno-unused-function")
    if(BUILD_WITH_MARCH_NATIVE)
      set(MY_FLAGS "${MY_FLAGS} -march=native")      
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "\n!Compiling in Debug mode!\n")
        # Debug
        set(MY_FLAGS "-Wall  -O3  -g") 
    endif()
endif()

# Add OpenMP flags
if( WITH_OPENMP )
    set(MY_FLAGS "${MY_FLAGS} -fopenmp")   
endif()

# In order to set RPATH (instead of RUNPATH) for g2o and other libraries linking issue.
# ISSUE: g2o binaries link to system g2o (here /opt/ros/noetic/lib/libg2o_core.so) and crash (different version); 
# we need to set LD_LIBRARY_PATH for correct linking
# https://stackoverflow.com/questions/47117443/dynamic-linking-with-rpath-not-working-under-ubuntu-17-10 
if(SET_RPATH)
  set(MY_FLAGS "${MY_FLAGS} -Wl,--disable-new-dtags")
endif()

# get and store all the active flags 
set(MY_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(MY_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")    

# apply flags 
set(CMAKE_C_FLAGS "${MY_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${MY_CXX_FLAGS}")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

#====================================
# cmake macros

include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

#====================================
# bin and lib destination 
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#====================================
# c++ macros 

set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Where data live")
add_compile_definitions(DATA_DIR=${DATA_DIR})

#====================================
# packages 

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/g2o/cmake_modules )

# fmt 
find_package(fmt REQUIRED)

# cholmod 
find_package(Cholmod REQUIRED)
#find_package(SuiteSparse REQUIRED)

# csparse 
find_package(CSparse REQUIRED)
message(STATUS "CSparse include: ${CSPARSE_INCLUDE_DIR}")

# eigen 
find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "eigen include: ${EIGEN3_INCLUDE_DIR}")

# boost 
find_package(Boost REQUIRED COMPONENTS thread system filesystem)

# pangolin 
#set(Pangolin_DIR ${PROJECT_SOURCE_DIR}/thirdparty/pangolin/build/) # this seems to work in general 
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/pangolin/install/lib/cmake/Pangolin/) # this does only work (after make install) if we set the RPATH!
set(Pangolin_DIR ${PROJECT_SOURCE_DIR}/thirdparty/pangolin/install/)  # this does only work (after make install) if we set the RPATH!
message(STATUS "set custom pangolin folder: ${Pangolin_DIR}")
find_package(Pangolin REQUIRED)
message(STATUS "pangolin include: ${Pangolin_INCLUDE_DIRS}")
message(STATUS "pangolin libs: ${Pangolin_LIBRARIES}")

# sophus 
# just including the headers (no build is needed)
set(Sophus_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/sophus)
message(STATUS "set custom sophus include folders: ${Sophus_INCLUDE_DIRS}")

# gtsam 
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/gtsam/install/lib/cmake/GTSAM/)
set(GTSAM_DIR ${PROJECT_SOURCE_DIR}/thirdparty/gtsam/install/)
message(STATUS "set custom gtsam folder: ${GTSAM_DIR}")
find_package(GTSAM REQUIRED)
message(STATUS "gtsam include: ${GTSAM_INCLUDE_DIR}")

# g2o 
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/g2o/install/lib/cmake/g2o/)
set(G2O_ROOT ${PROJECT_SOURCE_DIR}/thirdparty/g2o/install/)
#set(G2O_LIB_DIR ${PROJECT_SOURCE_DIR}/thirdparty/g2o/install/lib)
message(STATUS "set custom g2o root folder: ${G2O_ROOT}")
find_package(G2O REQUIRED)
message(STATUS "g2o include: ${G2O_INCLUDE_DIR}")
message(STATUS "g2o lid dir: ${G2O_LIB_DIR}")
message(STATUS "g2o libs: ${G2O_LIBS}")

# ceres 
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/ceres/install/lib/cmake/Ceres/)
set(Ceres_DIR ${PROJECT_SOURCE_DIR}/thirdparty/ceres/install/)
message(STATUS "set custom ceres folder: ${Ceres_DIR}")
find_package(Ceres REQUIRED)
message(STATUS "ceres include: ${CERES_LIBRARIES}")

# pcl 
find_package(PCL 1.8 REQUIRED)
add_definitions(-DPCL_NO_PRECOMPILE)  # this is strictly required when we define a new pcl point type; 
                                      # NOTE: apparently it solves a crash problem with test_pointcloud_mapping.cpp; unfortunately, it increases the build time
add_definitions( ${PCL_DEFINITIONS} )
message(STATUS "PCL_DEFINITIONS: ${PCL_DEFINITIONS}")

# opencv
message(STATUS "Desired OpenCV version: ${OPENCV_VERSION}")
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV install path: ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
include(CheckIncludeFileCXX) 
set (CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
check_include_file_cxx(opencv2/xfeatures2d.hpp HAVE_OPENCV_CONTRIB) # detect if we have opencv_contrib
if(HAVE_OPENCV_CONTRIB)
  message(STATUS "opencv_contrib found!")
  add_definitions(-DHAVE_OPENCV_CONTRIB)
endif()

# mahigui
set(MAHIGUI_DIR ${PROJECT_SOURCE_DIR}/thirdparty/mahigui/install/)
message(STATUS "set custom mahi-gui folder: ${MAHIGUI_DIR}")
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/mahigui/install/lib/cmake/mahi-gui/)
find_package(mahi-gui QUIET)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/mahigui/install/lib/cmake/mahi-util/)
find_package(mahi-util QUIET)

# json
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/json/install/share/cmake/nlohmann_json/)
set(JSON_DIR ${PROJECT_SOURCE_DIR}/thirdparty/json/install/)
message(STATUS "set custom json folder: ${JSON_DIR}")
find_package(nlohmann_json REQUIRED)

#sesync 
set(SESYNC_ROOT ${PROJECT_SOURCE_DIR}/thirdparty/sesync)
message(STATUS "set sesync root folder: ${SESYNC_ROOT}")
find_package(SESync REQUIRED)

#message(STATUS "sesync lid dir: ${SESYNC_LIB_DIR}")
message(STATUS "sesync libs: ${SESYNC_LIBS}")

# DBoW3
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/thirdparty/dbow3/install/lib/cmake/DBoW3)
set(DBOW3_DIR ${PROJECT_SOURCE_DIR}/thirdparty/dbow3/install/)
message(STATUS "set custom DBoW3 folder: ${DBOW3_DIR}")
find_package(DBoW3 REQUIRED)  
message(STATUS "DBoW3 includes: ${DBoW3_INCLUDE_DIRS}")

# ibow-lcd
set(OBINDEX2_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/include/)
set(OBINDEX2_LIB_DIR ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/build/)
set(IBOW_LCD_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/ibow-lcd/include/)
set(IBOW_LCD_LIB_DIR ${PROJECT_SOURCE_DIR}/thirdparty/ibow-lcd/build/)
set(IBOW_LCD_LIBS lcdetector obindex2)

# octomap 
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/octomap/;/usr/lib/")  # specify the system lib path otherwise you will find ROS octomap and it will bring in the ROS version of g2o (https://github.com/gaoxiang12/octomap_tutor/issues/3)
find_package(octomap REQUIRED)
message(STATUS "OCTOMAP_LIBRARIES: ${OCTOMAP_LIBRARIES}")

# glog
find_package(Glog REQUIRED)

# gtest
find_package(GTest REQUIRED)

# gflags
find_package(GFlags REQUIRED)


#====================================
# includes 

set(INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/utils/
)
# add all subdirs of utils 
subdirlist(SUBDIRS ${PROJECT_SOURCE_DIR}/utils) # macro from cmake/macros.cmake
foreach(subdir ${SUBDIRS})
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/utils/${subdir})
endforeach() 

set(EXT_INCLUDE_DIRS
  ${fmt_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${CHOLMOD_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${G2O_INCLUDE_DIR}  
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${SESYNC_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIRS}
  ${OBINDEX2_INCLUDE_DIR}
  ${IBOW_LCD_INCLUDE_DIR}
  ${OCTOMAP_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
)

#====================================
# libs 

# internal libraries built in the folder utils
set(LIBS
  orbslam2_features  
  bal_dataset 
)

link_directories( 
  ${G2O_LIB_DIR}  
  ${PCL_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${DBoW3_LIB_DIR}
  ${IBOW_LCD_LIB_DIR}
  ${OBINDEX2_LIB_DIR}
)

set(EXT_LIBS
  fmt::fmt 
  ${EIGEN3_LIBS}
  ${Boost_LIBRARIES} 
  ${CHOLMOD_LIBRARIES}  
  ${Pangolin_LIBRARIES}
  ${PCL_LIBRARIES} 
  gtsam
  ${G2O_LIBS}  
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
  mahi::gui  
  nlohmann_json::nlohmann_json
  ${SESYNC_LIBS}
  ${CSPARSE_LIBRARY}
  ${DBoW3_LIBS}
  ${IBOW_LCD_LIBS}
  ${OCTOMAP_LIBRARIES}
  ${GTEST_BOTH_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}  
)

#====================================
# source files and folders to build 

set(FOLDERS 
  utils
  algebra_geometry
  backend
  frontend
  io
  viz 
  loop_closure
  dense_mapping
  full_slam
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     