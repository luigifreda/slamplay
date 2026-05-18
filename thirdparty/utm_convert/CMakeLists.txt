cmake_minimum_required(VERSION 2.8)
project(utm_convert) 

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

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

#====================================

add_library(utm_convert
    utm.cc
    tranmerc.cc
)

target_include_directories(utm_convert
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

#====================================

# install
install(TARGETS utm_convert
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)

install(FILES
    utm.h
    tranmerc.h
    DESTINATION ${PROJECT_SOURCE_DIR}/include/utm_convert
)
