include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(timeutils TimeUtils.cpp TimeUtils.h)
target_link_libraries(timeutils ${CORELIBS})
