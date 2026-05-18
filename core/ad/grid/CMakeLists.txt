include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_grid 
        octo_tree.cpp octo_tree.h)
target_link_libraries(ad_grid ad_common ${CORELIBS})
