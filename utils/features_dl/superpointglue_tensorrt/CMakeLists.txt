include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(superpointglue SHARED
  SuperPoint.cpp
  SuperGlue.cpp
)
target_link_libraries(superpointglue ${CORELIBS})
