include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_io 
    io_utils.cpp io_utils.h
)
target_link_libraries(ad_io ad_common ${CORELIBS})
