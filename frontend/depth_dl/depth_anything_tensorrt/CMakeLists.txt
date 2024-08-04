include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(depth_anything_tensorrt SHARED
  depth_anything/DepthAnything.cpp
  depth_anything/DepthAnythingSettings.cpp
)
target_link_libraries(depth_anything_tensorrt ${CORELIBS})

add_executable(test_depth_estimation test_depth_estimation.cpp)
target_link_libraries(test_depth_estimation depth_anything_tensorrt ${CORELIBS})

