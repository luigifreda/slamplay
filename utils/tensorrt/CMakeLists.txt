include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(tensorrtutils STATIC
  tensorrt_utils.cpp
)
target_link_libraries(tensorrtutils ${CORELIBS})
