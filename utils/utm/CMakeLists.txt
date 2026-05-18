include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(utm_utils utm_utils.cpp utm_utils.h)
target_link_libraries(utm_utils ${CORELIBS} ${UTM_LIB_DIR}/libutm_convert.a)
