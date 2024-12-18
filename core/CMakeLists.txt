include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})


add_subdirectory(features)
add_subdirectory(features_dl)
add_subdirectory(depth_dl)