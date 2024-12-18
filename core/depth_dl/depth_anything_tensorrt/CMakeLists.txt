include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(depth_anything_tensorrt SHARED
  DepthAnything.cpp
  DepthAnythingSettings.cpp
)
target_link_libraries(depth_anything_tensorrt ${CORELIBS})
