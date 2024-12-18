cmake_minimum_required(VERSION 3.2)
project(ros_bridge_minimal) 

set(BUILD_WITH_MARCH_NATIVE ON  CACHE BOOL   "Build with \"-march native\"")

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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-unused-parameter -Wno-deprecated-declarations")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wno-deprecated-declarations")

if(BUILD_WITH_MARCH_NATIVE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -DNDEBUG")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3 -DNDEBUG")       
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-catch-value")
endif()

#====================================
# packages 

#find_package(catkin REQUIRED COMPONENTS cpp_common roscpp_serialization roscpp_traits rostime roslz4)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
find_package(BZip2 REQUIRED)
find_library(lz4_LIBRARIES NAMES lz4)

file(GLOB CONSOLE_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/console_bridge/src/*.cpp")
file(GLOB CPP_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/cpp_common/src/*.cpp")
file(GLOB ROSCPP_SERIALIZATION_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/roscpp_serialization/src/*.cpp")
file(GLOB ROSTIME_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/rostime/src/*.cpp")

file(GLOB ROSBAG_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/tools/rosbag_storage/src/*.cpp")
file(GLOB ROSLZ4_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/utilities/roslz4/src/[a-z]*.c")

message(STATUS "ROSLZ4_SRCS: ${ROSLZ4_SRCS}")

add_library(rosbag STATIC ${ROSBAG_SRCS} ${ROSTIME_SRCS} ${CPP_COMMON_SRCS} ${ROSCPP_SERIALIZATION_SRCS} ${ROSLZ4_SRCS} ${CONSOLE_BRIDGE_SRCS})

target_include_directories(rosbag PUBLIC
    include
    console_bridge/include
    roscpp_core/cpp_common/include
    roscpp_core/rostime/include
    roscpp_core/roscpp_serialization/include
    roscpp_core/roscpp_traits/include
    ros_comm/utilities/roslz4/include
    ros_comm/tools/rosbag_storage/include)

target_link_libraries(rosbag PUBLIC ${Boost_LIBRARIES} ${BZIP2_LIBRARIES} ${lz4_LIBRARIES})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)