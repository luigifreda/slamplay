include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")

add_executable(test_bundle_adjustment_g2o test_bundle_adjustment_g2o.cpp)
target_link_libraries(test_bundle_adjustment_g2o ${CORELIBS})

add_executable(test_bundle_adjustment_ceres test_bundle_adjustment_ceres.cpp)
target_link_libraries(test_bundle_adjustment_ceres ${CORELIBS})