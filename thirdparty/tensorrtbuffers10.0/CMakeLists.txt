cmake_minimum_required(VERSION 2.8)
#cmake_minimum_required(VERSION 3.16...3.27) # issues with opencv import 
project(tensorrtbuffers LANGUAGES CXX)

set(TensorRT_DIR  "${PROJECT_SOURCE_DIR}/../TensorRT" CACHE STRING "Desired TensorRT path") # this is the first target version that is searched for 


# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#====================================
# c++ standard 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # CMake will generate an error if it is unable to compile the code with a compiler that supports C++17.

#====================================

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)

# TensorRT
find_package(CUDA REQUIRED)
message(STATUS "CUDA_VERSION_STRING: ${CUDA_VERSION_STRING}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")    
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

message(STATUS "TensorRT_DIR: ${TensorRT_DIR}")
find_package(TensorRT REQUIRED)

message(STATUS "TensorRT_INCLUDE_DIR: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT_LIBRARIES: ${TensorRT_LIBRARIES}")   


#====================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(tensorrtbuffers SHARED 
  include/tensorrtbuffers/buffers.h
  include/tensorrtbuffers/common.h
  include/tensorrtbuffers/ErrorRecorder.h
  include/tensorrtbuffers/logger.h
  include/tensorrtbuffers/logging.h
  include/tensorrtbuffers/timingCache.h
  include/tensorrtbuffers/fileLock.h
  src/logger.cpp
  src/timingCache.cpp
  src/fileLock.cpp
)
target_link_libraries(tensorrtbuffers ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})


add_executable(test src/test.cpp)
target_link_libraries(test tensorrtbuffers)
