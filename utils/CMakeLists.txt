include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})


add_subdirectory(datasets)
add_subdirectory(time)
if(WITH_TENSORRT)
    add_subdirectory(tensorrt)
endif()