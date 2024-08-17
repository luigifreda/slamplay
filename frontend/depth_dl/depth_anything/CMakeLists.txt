include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

add_executable(test_depth_estimation test_depth_estimation.cpp)
target_link_libraries(test_depth_estimation depth_anything_tensorrt ${CORELIBS})

